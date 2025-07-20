# State.py

import numpy as np
import sympy
from typing import List, Tuple, Any
import scipy.special # import for lambdify to use

# ==============================================================================
# Setup & Constants
# ==============================================================================

_x = sympy.symbols('x')
erfc_numpy = sympy.lambdify(_x, sympy.erfc(_x), 'scipy')
erf_numpy = sympy.lambdify(_x, sympy.erf(_x), 'scipy')

# Define constants
K_RANGE_STEP = 0.1
K0S_RANGE_STEP = 51
EIGENVALUE_THRESHOLD = 0.98
SQRT_2 = np.sqrt(2)

# ==============================================================================
# State Class
# ==============================================================================

class State:
    """
    Represents and processes the state of the polarity analysis, including
    building and solving a Markov transition matrix.
    """
    def __init__(self, name: str):
        """
        Initializes the State object.

        Args:
            name (str): An identifier for this state instance.
        """
        self.name: str = name
        self.num: int = 0
        self.downthreshold: List[float] = []
        self.upthreshold: List[float] = []
        self.sample: List[Any] = []
        self.chances: List[Any] = []
        self.samplength: List[Any] = []
        self.xsquare: List[Any] = []
        self.pmi: List[float] = []
        self.Apeak: List[List[float]] = []
        self.combo: List[int] = []
        self.arrivaltimestamp: List[Any] = []
        self.multiplesolution: int = 1
        
        # Attributes to be calculated
        self.matrix: np.ndarray = np.array([])
        self.timeprob: List[np.ndarray] = []
        self.bigeig: np.ndarray = np.array([])
        self.eigvalue: np.ndarray = np.array([])
        self.ampprob_up: List[float] = []
        self.bestsigma: List[Any] = []
        self.polarityestimation: float = 0.0
        self.polarityunknown: float = 0.0
        self.polarityup: float = 0.0
        self.polaritydown: float = 0.0
        self.Apeakestimate: float = 0.0
        self.arrivalestimate: float = 0.0
        self.sigmaestimate: float = 0.0

    def addstate(self, downthreshold, upthreshold, noisesample, chances, pmi, Apeak, arrivaltime):
        """Adds a single (non-combo) state."""
        self.num += 1
        self.combo.append(1)
        self.downthreshold.append(downthreshold)
        self.upthreshold.append(upthreshold)
        self.sample.append(noisesample)
        self.chances.append(chances)
        self.samplength.append(len(noisesample))
        self.xsquare.append(np.sum(np.square(noisesample)))
        self.pmi.append(pmi)
        self.Apeak.append(Apeak)
        self.arrivaltimestamp.append(arrivaltime)

    def addcombo(self, combonum, downthreshold, upthreshold, noisesample, chances, pmi, Apeak, arrivaltime):
        """Adds a combined state from multiple sub-states."""
        self.num += 1
        self.combo.append(combonum)
        self.downthreshold.append(downthreshold)
        self.upthreshold.append(upthreshold)
        self.sample.append(noisesample)
        self.chances.append(chances)
        
        lengths = [len(ns) for ns in noisesample]
        squares = [np.sum(np.square(ns)) for ns in noisesample]
        
        self.samplength.append(lengths)
        self.xsquare.append(squares)
        self.pmi.append(pmi)
        self.Apeak.append(Apeak)
        self.arrivaltimestamp.append(arrivaltime)

    # --------------------------------------------------------------------------
    # Markov Matrix Calculation
    # --------------------------------------------------------------------------

    def _build_transition_matrix(self) -> np.ndarray:
        """Builds the transition matrix using vectorized operations."""
        matrix = np.zeros((self.num, self.num))
        down_th_arr = np.array(self.downthreshold)
        up_th_arr = np.array(self.upthreshold)

        for i in range(self.num):
            if i > 0 and self.samplength[i] == self.samplength[i-1] and np.all(self.sample[i] == self.sample[i-1]):
                matrix[i] = matrix[i-1]
                continue

            if self.combo[i] == 1:
                sigma = np.sqrt(self.xsquare[i] / self.samplength[i]) if self.samplength[i] > 0 else float('inf')
                p1 = erfc_numpy(down_th_arr / (SQRT_2 * sigma))
                p2 = erfc_numpy(up_th_arr / (SQRT_2 * sigma))
                p12 = np.exp(-self.chances[i][0] * p2 / 2) - np.exp(-self.chances[i][0] * p1 / 2)
            else:
                samplength_arr = np.array(self.samplength[i])
                xsquare_arr = np.array(self.xsquare[i])
                sigmas = np.sqrt(xsquare_arr / samplength_arr, where=samplength_arr>0)
                sigmas[samplength_arr==0] = float('inf')
                
                p1_matrix = erfc_numpy(down_th_arr[:, np.newaxis] / (SQRT_2 * sigmas))
                p2_matrix = erfc_numpy(up_th_arr[:, np.newaxis] / (SQRT_2 * sigmas))
                p12_matrix = np.exp(-self.chances[i][0] * p2_matrix / 2) - np.exp(-self.chances[i][0] * p1_matrix / 2)
                p12 = np.mean(p12_matrix, axis=1)
            
            row_sum = np.sum(p12)
            matrix[i] = p12 / row_sum if row_sum > 1e-9 else 0
        return matrix
        
    def _solve_for_orthogonal_basis(self, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Finds the best orthogonal basis vectors (probp, probs) for two eigenvectors."""
        k_range = np.arange(-50, 50 + K_RANGE_STEP, K_RANGE_STEP)
        dot_alpha00 = np.dot(alpha[0], alpha[0])
        dot_alpha01 = np.dot(alpha[0], alpha[1])
        dot_alpha11 = np.dot(alpha[1], alpha[1])

        k0p = k_range
        k1p = 1 - k0p
        k0sco = k0p * dot_alpha00 + k1p * dot_alpha01
        k1sco = k1p * dot_alpha11 + k0p * dot_alpha01
        
        with np.errstate(divide='ignore', invalid='ignore'):
            k0s = np.nan_to_num(-1 * k1sco / (k0sco - k1sco))

        k0s_search_grid = k0s[:, np.newaxis] + np.linspace(-1, 1, K0S_RANGE_STEP)[np.newaxis, :]
        probs_matrix = k0s_search_grid[..., np.newaxis] * alpha[0] + (1 - k0s_search_grid)[..., np.newaxis] * alpha[1]
        probp_matrix = k0p[:, np.newaxis] * alpha[0] + k1p[:, np.newaxis] * alpha[1]
        
        ortho_matrix = np.einsum('ij,ikj->ik', np.abs(probp_matrix), np.abs(probs_matrix))
        
        bestfit_per_k = np.min(ortho_matrix, axis=1)
        best_k_idx = np.argmin(bestfit_per_k)
        best_k0s_idx = np.argmin(ortho_matrix[best_k_idx])
        best_k0s = k0s_search_grid[best_k_idx, best_k0s_idx]
        
        k0p_best, k1p_best, k0s_best, k1s_best = k_range[best_k_idx], 1 - k_range[best_k_idx], best_k0s, 1 - best_k0s
        probp = k0p_best * alpha[0] + k1p_best * alpha[1]
        probs = k0s_best * alpha[0] + k1s_best * alpha[1]
        return probp, probs

    def markovmatrix(self) -> Tuple[List[np.ndarray], int]:
        """
        Main method to construct and solve the Markov transition matrix.
        """
        self.matrix = self._build_transition_matrix()
        eigenvalues, eigenvectors = np.linalg.eig(self.matrix.T)
        self.bigeig = np.around(np.sort(np.real(eigenvalues))[-5:][::-1], 3)
        
        eigindex = np.where(np.real(eigenvalues) > EIGENVALUE_THRESHOLD)[0]
        self.eigvalue = eigenvalues[eigindex]
        num_solutions = len(eigindex)

        if num_solutions == 1:
            eig_vec = eigenvectors[:, eigindex[0]]
            timeprob = np.real(eig_vec / np.sum(eig_vec)).flatten()
            self.timeprob = [np.abs(timeprob)]
        elif num_solutions >= 2:
            alpha = np.zeros((2, self.num))
            for i in range(2):
                eig_vec = eigenvectors[:, eigindex[i]]
                alpha[i] = np.real(np.abs(eig_vec) / np.sum(np.abs(eig_vec))).flatten()
            probp, probs = self._solve_for_orthogonal_basis(alpha)
            self.timeprob = [np.abs(probp), np.abs(probs)]
        else: # num_solutions == 0
            self.timeprob = []
        
        return self.timeprob, num_solutions

    def ampprobcalculate(self) -> np.ndarray:
        """
        Calculates the probability of an 'up' polarity based on amplitude.
        """
        self.ampprob_up = []
        self.bestsigma = []
        # Convert to numpy arrays for vectorized operations
        apeak_arr = np.array(self.Apeak, dtype=object)

        for i in range(self.num):
            if self.combo[i] == 1:
                sigma = np.sqrt(self.xsquare[i] / self.samplength[i])
                prob_up = 0.5 + 0.5 * erf_numpy(apeak_arr[i][0] / (SQRT_2 * sigma))
                self.bestsigma.append(sigma)
            else:
                sigmas = np.sqrt(np.array(self.xsquare[i]) / np.array(self.samplength[i]))
                probs_up = 0.5 + 0.5 * erf_numpy(apeak_arr[i] / (SQRT_2 * sigmas))
                prob_up = np.mean(probs_up)
                self.bestsigma.append(sigmas)
            
            self.ampprob_up.append(np.clip(prob_up, 0, 1))

        return np.array(self.ampprob_up)

    def estimation(self, qualified_id: int):
        """
        Calculates final estimated parameters based on a qualified solution.
        """
        timeprob = self.timeprob[qualified_id]
        ampprob_up_arr = np.array(self.ampprob_up)
        
        self.polarityestimation = np.sum(timeprob * ampprob_up_arr)
        Apeak_arr = np.array([item[0] for item in self.Apeak])
        unknownindex = np.where(Apeak_arr == 0)[0]
        knownindex = np.where(Apeak_arr != 0)[0]
        
        self.polarityunknown = np.sum(timeprob[unknownindex])
        self.polarityup = np.sum(timeprob[knownindex] * ampprob_up_arr[knownindex])
        self.polaritydown = 1 - self.polarityup - self.polarityunknown
        Apeakestimate_vec = np.array([np.mean(p) for p in self.Apeak])
        arrivalestimate_vec = np.array([np.mean(t) for t in self.arrivaltimestamp])
        sigmaestimate_vec = np.array([np.mean(s) for s in self.bestsigma])

        self.Apeakestimate = np.sum(timeprob * Apeakestimate_vec)
        self.arrivalestimate = np.sum(timeprob * arrivalestimate_vec)
        self.sigmaestimate = np.sum(timeprob * sigmaestimate_vec)

    def getstateinform(self, stateid: int) -> Tuple:
        """Retrieves information for a given state ID."""
        if not (0 <= stateid < self.num):
            print('wrong stateid')
            return -1, -1, -1, -1, -1, -1
        
        return (
            self.combo[stateid], self.downthreshold[stateid],
            self.upthreshold[stateid], self.sample[stateid],
            self.pmi[stateid], self.Apeak[stateid]
        )

    def getstateprob(self, qualifiedid: int, stateid: int) -> Tuple[float, float]:
        """Retrieves probability for a given state ID and qualified solution."""
        if not (0 <= stateid < self.num and 0 <= qualifiedid < len(self.timeprob)):
            print('wrong stateid or qualifiedid')
            return -1, -1

        return self.timeprob[qualifiedid][stateid], self.ampprob_up[stateid]