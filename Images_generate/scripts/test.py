def binary_x(x, threshold=0.5):
    return (x > threshold).float()


binary_x(0.3)

input_name = '../weights/blood_50k.pth'
output_name = '../images/'+input_name.split('.')[-2].split('/')[-1]+'_image.gif'

print("Step:"+str(i)+"/"+str(num_iterations))

if (i + 1) % save_interval == 0 or i==0:
