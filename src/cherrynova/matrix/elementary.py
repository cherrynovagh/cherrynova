#Reusable functions
import torch

#Row swap
def rowswap(matrix, row1, row2):
    new_matrix = matrix.clone()
    new_matrix[[row1, row2]] = new_matrix[[row2, row1]]
    return new_matrix

#Row scale
def rowscale(matrix, row, scalar):
    new_matrix = matrix.clone()
    new_matrix[row] = new_matrix[row] * scalar
    return new_matrix

#Row replacement
def rowreplacement(matrix, row_i, row_j, j, k):
    new_matrix = matrix.clone()
    new_matrix[row_i] = j * matrix[row_i] + k * matrix[row_j]
    return new_matrix

#Reduced Row Echelon Form
def rref(matrix):
    new_matrix = matrix.clone()
    rows, cols = new_matrix.shape
    pivot_row = 0
    for col in range(cols):
        if pivot_row >= rows:
            break
        if new_matrix[pivot_row, col] == 0:
            for r in range(pivot_row + 1, rows):
                if new_matrix[r, col] != 0:
                    new_matrix = rowswap(new_matrix, pivot_row, r)
                    break
            else:
                continue
        pivot_value = new_matrix[pivot_row, col]
        new_matrix = rowscale(new_matrix, pivot_row, 1 / pivot_value)
        for r in range(pivot_row + 1, rows):
            factor = new_matrix[r, col]
            if factor != 0:
                new_matrix = rowreplacement(new_matrix, r, pivot_row, 1, -factor)
        pivot_row += 1
    return new_matrix

#Inputting the data
if __name__ == "__main__":
    M = torch.tensor([[1, 3, 0, 0, 3], [0, 0, 1, 0, 9], [0, 0, 0, 1, -4]], dtype=torch.float32)
    print("Original Matrix:")
    print(M)
    step1 = rowswap(M, 0, 1)
    print("Row Swap:")
    print(step1)
    step2 = rowscale(step1, 1, 1/3)
    print("Row Scale:")
    print(step2)
    step3 = rowreplacement(step2, 2, 0, 1, -3)
    print("Row Replacement:")
    print(step3)
    rref_result = rref(M)
    print("RREF:")
    print(rref_result)