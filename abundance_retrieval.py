import os
import argparse
import random
import time
import numpy as np
import pyro
import pyro.distributions as dist
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from pyro.infer import MCMC, NUTS
from tqdm import tqdm

from hmc_epistemic_model import FullSEGNNModel
from dataset_QM9 import getQM9Data
from balanced_irreps import BalancedIrreps
from e3nn.o3 import Irreps

def set_all_seeds(num):
    random.seed(num)
    np.random.seed(num)
    torch.manual_seed(num)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(num)


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    main_path = args.main_path

    set_all_seeds(42)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    PATH = f'./retrieval'
    os.makedirs(PATH, exist_ok=True)
    #save_to_text(f'{PATH}/args.txt', args.__dict__)

    data_size = args.data_size
    test_size = args.test_size
    ordered_indices = range(0, 129816 + 1)

    mol_pos_npz = np.load(f'{main_path}/mol_pos.npz', allow_pickle=True)
    mol_idx_npz = np.load(f'{main_path}/mol_idx.npz', allow_pickle=True)
    mol_pos_list = mol_pos_npz[mol_pos_npz.files[0]]
    mol_idx_list = mol_idx_npz[mol_idx_npz.files[0]]
    mol_list = np.load(f'{main_path}/mol_list.npy')
    mol_smile_names = np.load(f'{main_path}/mol_name.npy', allow_pickle=True)

    spectra_data = getQM9Data(mol_pos_list, mol_idx_list, mol_list, lmax_attr=args.lmax_attr, wave_eng=args.wave_eng, data_idx=ordered_indices, main_path=args.main_path)
    test_indices = [102649, 109851, 20029, 39528, 61471, 92732, 102882, 111224, 23976, 41921, 71155, 98118, 105308, 11320, 27256, 44322, 73703, 107093, 116357, 32290, 4568, 78691, 108701, 12285, 37033, 45710, 79977,109317, 128598, 3887, 59918, 87273]
    for idx in tqdm(test_indices):
        atom_tensor = torch.tensor(mol_idx_list[idx], device=device)
        atom_positions = torch.tensor(mol_pos_list[idx], device=device)
        processed_data = spectra_data.get_single_molecule_data(atom_tensor, atom_positions, T=0, P=0).to(device)

    input_irreps = Irreps(f"{18 + 1 + 2 + 0}x0e") # 18 one-hot, 1 mass, 3 global
    output_irreps = Irreps("1x0e")

    edge_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
    node_attr_irreps = Irreps.spherical_harmonics(args.lmax_attr)
    hidden_irreps = BalancedIrreps(args.lmax_h, args.hidden_features, True)
    model = FullSEGNNModel(input_irreps, hidden_irreps, output_irreps, edge_attr_irreps, node_attr_irreps, norm="batch", num_layers=args.num_layers)
    model = model.to(device)
    model_checkpoint = '/projects/mccleary_group/berman.ed/biomarker_v2/Biomarker/results/QM9_129784_20250317-183502/epoch_199.pt'
    model.load_state_dict(torch.load(model_checkpoint, map_location=device))

    random_molecule_one = random.choice(test_indices)
    print(f"Random molecule one: {random_molecule_one}")
    atomic_positions_one = torch.tensor(mol_pos_list[random_molecule_one], device=device)
    atomic_tensor_one = torch.tensor(mol_idx_list[random_molecule_one], device=device)
    processed_atom_one = spectra_data.get_single_molecule_data(atomic_tensor_one, atomic_positions_one, T=0, P=0).to(device)

    spectral_one, _, ep_variance_one, _, _, _, _ = model(processed_atom_one)
    spectral_one = spectral_one[0]
    spectral_one = spectral_one.squeeze(0)
    spectral_one = spectral_one.to(device)
    ep_variance_one = ep_variance_one.squeeze(0).to(device)

    random_molecule_two = random.choice(test_indices)
    print(f"Random molecule two: {random_molecule_two}")
    atomic_positions_two = torch.tensor(mol_pos_list[random_molecule_two], device=device)
    atomic_tensor_two = torch.tensor(mol_idx_list[random_molecule_two], device=device)
    processed_atom_two = spectra_data.get_single_molecule_data(atomic_tensor_two, atomic_positions_two, T=0, P=0).to(device)

    spectral_two, _, ep_variance_two, _, _, _, _ = model(processed_atom_two)
    spectral_two = spectral_two[0]
    spectral_two = spectral_two.squeeze(0)
    spectral_two = spectral_two.to(device)
    ep_variance_two = ep_variance_two.squeeze(0).to(device)
    
    del mol_list, mol_pos_list, mol_idx_list

    abundances = torch.tensor([0.7, 0.3], device=device)
    abundances_matrix = torch.diag(abundances)
    spectra_matrix = torch.stack((spectral_one, spectral_two), dim=0)
    ep_variance_matrix = torch.stack((ep_variance_one, ep_variance_two), dim=0)
    combined_spectra_truth = torch.matmul(abundances_matrix, spectra_matrix).sum(dim=0)

    num_atoms = abundances.shape[0]

    def pyro_model(spectra_matrix=spectra_matrix, ep_variance_matrix=ep_variance_matrix, combined_spectra_truth=combined_spectra_truth, num_atoms=num_atoms, abundances_matrix=abundances_matrix):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        abundance_lows_prior = torch.zeros(num_atoms, device=device)
        abundance_highs_prior = torch.ones(num_atoms, device=device)
        abundances = pyro.sample("abundances", dist.Uniform(abundance_lows_prior, abundance_highs_prior))

        abundances_matrix = abundances_matrix.to(device) 
        spectra_matrix = spectra_matrix.to(device)
        uncertainty_matrix = ep_variance_matrix.to(device)

        combined_spectra = torch.matmul(abundances_matrix, spectra_matrix).sum(dim=0)
        combined_variance = torch.matmul(abundances_matrix, uncertainty_matrix).sum(dim=0)

        spectral_cov = (
            torch.diag(combined_variance)
            + 1e-6 * torch.eye(combined_variance.shape[0]).to(device)  # Regularization
        )

        combined_spectra_truth = combined_spectra_truth.to(device)

        pyro.sample(
            "mixed_spectra",
            dist.MultivariateNormal(combined_spectra, covariance_matrix=spectral_cov),
            obs=combined_spectra_truth,
        )

    nuts_kernel = NUTS(pyro_model, step_size=0.005, adapt_step_size=True)
    mcmc = MCMC(nuts_kernel, num_samples=1000, warmup_steps=1000)
    mcmc.run()
    samples = mcmc.get_samples()

    abundances_samples = samples["abundances"]
    abundance_samples_one = abundances_samples[:, 0]
    abundance_samples_two = abundances_samples[:, 1]

    abundance_samples_one = abundance_samples_one.cpu().numpy()
    abundance_samples_two = abundance_samples_two.cpu().numpy()

    np.save(f"./retrieval/abundance_1000_samples_one.npy", abundance_samples_one)
    np.save(f"./retrieval/abundance_1000_samples_two.npy", abundance_samples_two)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lmax_attr', type=int, help='Irreps max l', default=2)
    parser.add_argument('--lmax_h', type=int, help='hidden featrues max l', default=2)
    parser.add_argument('--batch_size', type=int, help='batch size', default=512)
    parser.add_argument('--hidden_features', type=int, help='number of hidden features', default=1024)
    parser.add_argument('--lr', type=float, help='learning rate', default=1e-3)
    parser.add_argument('--num_layers', type=int, help='number of layers', default=3)
    parser.add_argument('--epochs', type=int, help='number of epochs', default=100)
    parser.add_argument('--wave_eng', type=str, help='how to transform wave', default='min_max')
    parser.add_argument('--data_size', type=int, help='number of molecules to train', default=1000)
    parser.add_argument('--test_size', type=int, help='number of molecules to test', default=10)
    parser.add_argument('--main_path', type=str, help='path to the data', default='..')

    args = parser.parse_args()

    main(args)
