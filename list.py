import os as os


file_write = open("labels.txt", "w") 

for file_name in os.listdir("C:/Users/chann/major_project/augmented_databook"):
    file_write.write(f"{file_name}\n")

file_write.close()