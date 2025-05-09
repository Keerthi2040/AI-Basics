import streamlit as st
import torch

st.title("PyTorch and Streamlit Test")

# Test PyTorch functionality
tensor = torch.tensor([1, 2, 3])
st.write("PyTorch Tensor:", tensor)