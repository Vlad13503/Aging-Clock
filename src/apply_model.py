from preprocessing import load_celltype_data

cell_type = "erythrocyte"
df = load_celltype_data("../data/AIDA.h5ad", cell_type)

print(df)
