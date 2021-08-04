import os

nnunet_path = os.path.dirname(os.path.abspath(__file__))
for i in range(1):
    nnunet_path = os.path.dirname(nnunet_path)

print(f"nnunet_path: {nnunet_path}")
