#Code for Simulator 
def calculate_total_yield(M):
    """
    input M : list of list of size n * n
    output total_yield : total yield of crops in the field 
    """
    total_yield = 0
    for i in range(len(M[0])):
        for j in range(len(M[0])):
            # Case 1 : if there is a corn crop in the i,j cell of the field
            if M[i][j] == 'c':
                # Following code checks the number of bean crops adjacent to the corn crop 
                adjacent_beans = 0
                k = i-1
                l = j-1
                while k <= i+1:
                    if k >= 0 and k < len(M[0]) :
                        while l <= j+1:
                            if l >= 0 and l < len(M[0]):
                                if M[k][l] == 'b':
                                    adjacent_beans += 1
                            l += 1
                    l = j-1
                    k += 1
                total_yield += 10 + adjacent_beans
            
            # Case 2 : if there is a bean crop in the i,j cell of the field
            if M[i][j] == 'b':
                # Following code checks the number of corn crops adjacent to the bean crop 
                adjacent_corns = 0
                k = i-1
                l = j-1
                while k <= i+1:
                    if k >= 0 and k < len(M[0]):
                        while l <= j+1:
                            if l >= 0 and l < len(M[0]):
                                if M[k][l] == 'c':
                                    adjacent_corns += 1
                            l += 1
                    l = j-1
                    k += 1
                if adjacent_corns > 0:
                    total_yield += 15
                else:
                    total_yield += 10
            #print (i,j,M[i][j],total_yield)
    return total_yield

M=[['-','b','b'],
  ['-','c','b'],
  ['-','c','-']]
print("Total yield of the field is ",calculate_total_yield(M)) 