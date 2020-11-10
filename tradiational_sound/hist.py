import matplotlib.pyplot as plt  

methods=["Fbanks(SVM)","MFCC(SVM","LPC(Only)","LPC(SVM)","PLP(SVM)","RPLP(SVM)"]

values =[0.99,0.99,0.17,0.94,0.96,0.34]

plt.bar(methods,values,color='rgb')
plt.title("Different Methods on Formant classification")
plt.xlabel("Methods")
plt.ylabel("accuracy")
for x,y in enumerate(values):
    print(x,y)
    plt.text(x,y,y,ha='center',fontsize=16,color='blue')
plt.savefig("bars",format='png')
plt.show()