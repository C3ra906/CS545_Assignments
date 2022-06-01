#Cera Oh. CS545 ML. Prog 3. Winter 2022
import numpy as np
import random
from matplotlib import pyplot as plt

#Data Path
path = './545_cluster_dataset programming 3.txt'

#Function that loads the data file
def load_data(file_path):
    with open (file_path, 'r') as data:
        data = np.genfromtxt(file_path, delimiter='  ')
    return data

############################ K-MEANS ##################################
#Function to randomly select k data points as initial means
def initial_k(k, data):
    k_means = np.random.permutation(data)[:k] #picks k random points from the data set as initial means
    print(f"initial chosen: \n {k_means}")

    return k_means

#Function to assign each point to closest mean
def assignment(k, k_means, data):
    counter, index = 0, 0
    member = np.zeros(k) #array to keep track of how many points belong to a cluster
    dist = np.zeros(k) #array to keep track of Euclidiean distance
    cluster = np.zeros(1500) #array to keep track of clusters assigned for points

    for row in data:
        for k in k_means:
           dist[counter] = np.sqrt(np.square(k[0] - row[0]) + np.square(k[1] - row[1])) #Euclidean distance bewteen current point and a mean
           counter += 1
        counter = 0
        minimum = np.argmin(dist) #Find index of min distance
        cluster[index] = minimum
        member[minimum] += 1
        index += 1

    return cluster, member

#Function to recompute means
def recompute(k, data, cluster, member):
    add = np.zeros((k,2))
    new_k_means = np.zeros((k,2)) #array to hold newly calculated means

    counter = 0
    for pt in cluster:
        index = int(pt)
        add[index,0] += data[counter,0]
        add[index,1] += data[counter,1]
        counter += 1

    counter = 0
    for a in add:
        if member[counter] != 0:
            a = a/member[counter]
            new_k_means[counter,0] = a[0]
            new_k_means[counter,1] = a[1]
        counter+= 1

    return new_k_means

#K-Means Algorithm
def k_means_alg(k, k_means, data):
    old_k = np.copy(k_means)

    cluster, member = assignment(k, k_means, data)
    new_k_means = recompute(k, data, cluster, member)

    #Stopping Condition
    comparison = old_k == new_k_means
    equal = comparison.all()

    if (equal == False):
        new_k_means, cluster, E = k_means_alg(k, new_k_means, data) #Recursively run K-Means Algorithm
    else:
        print("K_means finished. The final centroids are:")
        print(new_k_means)
        E = SSE(k, new_k_means, cluster, data)
        print(f"SSE: {E}")

        #For Grpahing
        plot = input("Enter 1 to print:\n")
        plot = int(plot)
        if plot == 1:
            plotter(k, new_k_means, cluster, data)

    return new_k_means, cluster, E

#Function to run K-Means r times from r differently chosen initalizations and find lowest SSE
def run_k_means(data):
    run = 0
    r = input("Please enter the number of times to run K-Means:\n")
    r = int(r)
    k = input("Please enter the number of clusters:\n")
    k = int(k)    

    lowest_SSE = np.zeros(r) #array to keep track of SSE values
    best_cluster = np.zeros ((r,k,2)) #array to keep track of final means

    while (r != 0):
        print(f"Run {run + 1}:")
        k_means = initial_k(k, data)
        final, cluster, E = k_means_alg(k, k_means, data)
        lowest_SSE[run] = E
        best_cluster[run] = final
        r -= 1
        run += 1

    minimum = np.argmin(lowest_SSE) #Find index min distance
    print(f"Lowest SSE was run {minimum + 1} with value {lowest_SSE[minimum]} with clusters:")
    print(f"{best_cluster[minimum]}") 
    
#Function to calculate Sum of Square Error for K-Means
def SSE(k, k_means, cluster, data):
    rss = np.zeros(k)
    counter = 0
    E = 0

    for row in data:
        c = cluster[counter]
        c = int(c)
        rss[c] += np.square((k_means[c,0] - row[0]) + (k_means[c,1] - row[1]))
        counter += 1
    for index in rss:
        E += index

    return E

#Function to plot K-Means clusters
def plotter(k, k_means, cluster, data):
    colors = np.zeros((k, 3)) #array to hold colors

    #Assign random colors per cluster
    counter = 0
    while counter < k:
        r = random.random()
        b = random.random()
        g = random.random()
        colors[counter,0] = r
        colors[counter,1] = b
        colors[counter,2] = g
        counter += 1

    #Plot points
    counter = 0
    for row in data:
        x = row[0]
        y = row[1]
        index = int(cluster[counter])
        r = colors[index, 0]
        b = colors[index, 1]
        g = colors[index, 2]
        color = (r,b,g)
        plt.scatter(x, y, c=np.array([color]))
        counter += 1

    #Plot means
    c_x, c_y = k_means.T
    plt.scatter(c_x, c_y, c='k')

    plt.show()

############################ FUZZY C-MEANS ##################################

#Function to set initial membership grades
def initial_fcm(c, m, data):    
    grades = np.zeros((1500,c)) #initalize grade matrix

    #Assign random grades between 0.0-1.0
    for row in grades:
        counter = 0
        num = 100
        val = 0
        while counter < (c - 1):
            if num != 0:
                row[counter] = random.randint(0, num)/100
                val += int(row[counter] * 100)
                num -= int(row[counter] * 100)
            else:
                row[counter] = 0
            counter += 1
        row[counter] = (100 - val)/100

    return grades

#Function to calculate centroids
def centroid_finder(c, m, data, centroids, grades):  
    c_index = 0
    denom = np.zeros(c)

    for c in centroids:
        counter = 0
        for row in data:
            c[0] += row[0] * np.power(grades[counter, c_index], m)
            c[1] += row[1] * np.power(grades[counter, c_index], m)
            denom[c_index] += np.power(grades[counter, c_index],m)
            counter += 1 
        if denom[c_index] != 0:
            c[0] /= denom[c_index]
            c[1] /= denom[c_index] 
        else: #Prevent division by 0
             print("Dem = 0 in centroid finder")
             c[0] /= 0.000000001
             c[1] /= 0.000000001
        c_index += 1

    return centroids

#Function to recompute grades
def recompute_w(c, m, data, centroids, grades):
    row_count = 0    

    p = 2/(m - 1)

    for row in data:        
        index = 0
        while index < c:    
            add = 0
            counter = 0
            num = np.sqrt(np.square(row[0] - centroids[index][0]) + np.square(row[1] - centroids[index][1]))
            while counter < c:
                den = np.sqrt(np.square(row[0] - centroids[counter][0]) + np.square(row[1] - centroids[counter][1]))
                if den != 0:
                    add += np.power((num/den),p)
                elif den == float(inf): #Prevent division by inf
                     print("den is inf")
                     add += 0
                else:
                    print("den = 0 in recompute w") #Prevent division by 0
                    add += np.power((num/0.000000001), p)
                counter += 1
            if add != 0:
                grades[row_count][index] = 1/add
            else:
                 print("add = 0 in recompute w") #Prevent division by 0
                 grades[row_count][index] = 1/0.000000001          
            index+= 1         
        row_count += 1

    return grades

#Function Fuzzy C-Means
def fcm_alg(c, m, data, centroids, grades):    
    new_centroids = np.zeros((c,2))
    new_grades = np.zeros((1500, c))

    old_centroids = np.copy(centroids)

    new_centroids = centroid_finder(c, m, data, centroids, grades)
    new_grades = recompute_w(c, m, data, new_centroids, grades)

    #Round to 4th decimial place for stopping condition check
    old_centroids = np.round(old_centroids, 4)
    new_centroids = np.round(new_centroids, 4)

    #Stopping Condition
    comparison = new_centroids == old_centroids
    equal = comparison.all()

    if (equal == False):
        new_centroids, new_grades, E = fcm_alg(c, m, data, new_centroids, new_grades)
    else:
        print("Funzzy C-means finished. The final centroids are:")
        print(new_centroids)
        E = fuzzy_SSE(c, data, new_centroids, new_grades)
        print(f"SSE: {E}")
        
        #For Graphing
        plot = input("Enter 1 to print:\n")
        plot = int(plot)
        if plot == 1:
            fuzzy_plotter(c, data, new_centroids, new_grades)

    return new_centroids, new_grades, E

#Function to run Fuzzy C-Means r times from r differently chosen initalizations and finds lowest SSE
def run_fcm(data):
    m = 2
    run = 0

    r = input("Please enter the number of times to run Fuzzy C-Means:\n")
    r = int(r)
    c = input("Please enter the number of clusters:\n")
    c = int(c)    

    lowest_SSE = np.zeros(r) #array to keep track of SSE values
    best_cluster = np.zeros ((r,c,2)) #array to keep track of final centroids
    
    while (r != 0):
        print(f"Run {run + 1}:")
        grades = initial_fcm(c, m, data)
        centroids = np.zeros((c,2))#initalize array to hold centroids
        final, new_grades, E = fcm_alg(c, m, data, centroids, grades)
        lowest_SSE[run] = E
        best_cluster[run] = final
        r -= 1
        run += 1

    minimum = np.argmin(lowest_SSE) #Find index min SSE
    print(f"Lowest SSE was run {minimum + 1} with value {lowest_SSE[minimum]} with centroids:")
    print(f"{best_cluster[minimum]}") 
    
#Function to calculate Fuzzy C-Means Sum of Square Error
def fuzzy_SSE(c, data, centroids, grades):
    rss = np.zeros(c)
    counter = 0
    E = 0

    for row in data:
        m = np.argmax(grades[counter,]) #Find index max grades
        m = int(m)
        rss[m] += np.square((centroids[m,0] - row[0]) + (centroids[m,1] - row[1]))
        counter += 1
    for index in rss:
        E += index

    return E

#Function to plot Fuzzy C-Means Clusters
def fuzzy_plotter(c, data, centroids, grades):
    colors = np.zeros((c, 3)) #array to hold colors

    #Assign random colors to clusters
    counter = 0
    while counter < c:
        r = random.random()
        b = random.random()
        g = random.random()
        colors[counter,0] = r
        colors[counter,1] = b
        colors[counter,2] = g
        counter += 1

    #Assign highest grade cluster color to a point and plot
    counter = 0
    for row in data:
        x = row[0]
        y = row[1]
        index = np.argmax(grades[counter,]) #Find index max grades
        index = int(index)
        r = colors[index, 0]
        b = colors[index, 1]
        g = colors[index, 2]
        color = (r,b,g)
        plt.scatter(x, y, c=np.array([color]))
        counter += 1

    #Plot centroids
    c_x, c_y = centroids.T
    plt.scatter(c_x, c_y, c='k')

    plt.show()

def main():
    data_set = load_data(path)
    run_k_means(data_set)
    run_fcm(data_set)

if __name__ == '__main__':
    main()
