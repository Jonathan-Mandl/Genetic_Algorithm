#Bio - informatics, Ex2
#Ariel Barmats 314970054
#Yonatan Mandl 211399175

import numpy as np
import random
import copy
import os

k = 1
population_size = 100*k
permutation_size = 26
letters_idx_map = {chr(65 + i): i for i in range(26)} # a map of every letter in english to its index from 0 to 25
letters = letters_idx_map.keys() # list of letters in english
letter_frequency_1 = {} # frequency of letters from Letter_Freq
letter_frequency_2 = {} # frequency of letters from Letter2_Freq
dictionary = set()
generation = 0 # global variable - represents the generation number
total_fitness_calls=0
steps = 0

# sign of the algorithm type
regular = "regular"
darwin = "darwin"
lamark = "lamark"


def initialize():
    """
    creates matrix with 100 rows and 26 columns. Each row is a random permutation of 26 letters in English.
    :return: matrix
    """
    letters_set = set(letters)
    letters_list = list(letters_set)
    # initialize empty matrix
    matrix = np.empty((population_size, permutation_size), dtype='U1')

    for i in range(population_size):
        # Generate a random permutation
        random_permutation = random.sample(letters_list, len(letters_list))
        # add permutation to matrix row
        matrix[i, :] = random_permutation

    return matrix


def read_file():
    """
    reads file of encryption.
    :return:  file content
    """
    with open("enc.txt") as file:
        content = file.read()
    return content

def read_dictionary():
    """
    reads dictionary from file
    :return: set of all words in the dictionary.
    """
    with open("dict.txt") as file:
        for line in file.readlines():
            if line == '\n':
                continue
            dictionary.add(line[:-1])


def count_existing_words(encryption):
    """
    counts the numbers of words in the test that don't appear in the dictionary. This is a measure of the fitness of a solution
    :param encryption: the encryption of the text according to a permutation.
    :return: fitness - number of words in the test that don't appear in the dictionary
    """
    fitness = 0
    encryption = encryption.upper()
    for letter in encryption:
        # check if character is not one of the alphabet letters
        if ord(letter) > 90 or ord(letter) < 65:
            # if character is not an english letter, replace in with space
            encryption.replace(letter, " ")
    encryption = encryption.lower()
    # words is a list of all words in encryption
    words = encryption.split()
    # add one for every word not in dictionary
    for word in words:
        if word not in dictionary:
            fitness += 1
    return fitness


def replace_letters(file_content, permutation):
    """
    :param file_content: the content of encryption file
    :param permutation: a permutation represents a possible solution
    :return: ancrypted text according to permutation
    """
    encrypted_text = file_content.lower()
    # iterate over english letters
    for letter in letters:
        low_letter = letter.lower()
        # replace english letter in encryption with the letter that it is mapped to in the permutation.
        encrypted_text = encrypted_text.replace(low_letter, permutation[letters_idx_map[letter]])
    encrypted_text = encrypted_text.lower()

    return encrypted_text


def find_fitness(encryption):
    global total_fitness_calls
    """
    :param encryption: the encryption of text according to the permutation
    :return: the fitness score based on the sum of 3 fitness functions: number of words not in dictionary, difference from letter frequency 1 and difference from letter frequency 2
    """
    total_fitness_calls+=1
    fitness = 0
    fitness += compare_letter_frequency1(encryption) * 800
    fitness += compare_letter_frequency2(encryption) * 1200
    fitness += count_existing_words(encryption)
    return fitness


def compare_letter_frequency1(encryption):
    """
    :param encryption: the encryption of text according to the permutation
    :return:  difference of the frequency of letter in encryption from the letter frequency 1 from file
    """
    fitness = 0
    # mapping of every letter in english to its current frequency in encryption. initialized to 0
    current_frequency = {chr(65 + i): 0 for i in range(26)}
    letter_counter = 0
    encryption = encryption.upper()
    for letter in encryption:
        # check if letter is an english letter
        if 65 <= ord(letter) <= 90:
            # add 1 to current frequency of letter
            current_frequency[letter] += 1
            letter_counter += 1
    for letter in current_frequency.keys():
        # divide letter frequency of each letter by the total letters in encryption to get the fraction of letter frequency
        current_frequency[letter] /= letter_counter
    for letter in current_frequency.keys():
        # calculate  the difference of the frequency of letter in encryption from the frequency of letter in letter frequency 1 from file. add to fitness
        fitness += abs(current_frequency[letter] - letter_frequency_1[letter])
    return fitness


def compare_letter_frequency2(encryption):
    """
    :param encryption: the encryption of text according to the permutation
    :return:  difference of the frequency of letter pairs in encryption from the letter frequency 2 form file
    """
    fitness = 0
    # mapping of every letter pair in english to its current frequency in encryption. initialized to 0
    current_frequency = {chr(65 + i) + chr(65 + j): 0 for i in range(26) for j in range(26)}
    #counter of all letter pairs read so far
    letter_pair_counter = 0
    encryption = encryption.upper()
    idx = 0
    for letter in encryption:
        if idx + 1 == len(encryption):
            break
        # check if letter is an english letter
        if 65 <= ord(letter) <= 90:
            next_letter = encryption[idx + 1]
            # check if next letter is an english letter
            if 65 <= ord(next_letter) <= 90:
                # add 1 to current frequency of letter pair
                current_frequency[letter + next_letter] += 1
                letter_pair_counter += 1
        idx += 1
    for pair in current_frequency.keys():
        # divide letter frequency of each letter pair by the total letters in encryption to get the fraction of letter frequency
        current_frequency[pair] /= letter_pair_counter
    for pair in current_frequency.keys():
        # calculate  the difference of the frequency of letter pair in encryption from the frequency of letter pair in letter frequency 2 from file. add to fitness
        fitness += abs(current_frequency[pair] - letter_frequency_2[pair])

    return fitness


def read_letter_frequency():
    """
    reads letter frequency from Letter_Freq file and Letter2_Freq file. reads the frequency into global letter_frequency_1 and letter_frequency_2 dictionaries
    :return:
    """
    with open("Letter_Freq.txt") as file:
        for line in file.readlines():
            frequency, letter = line.split("\t")
            letter_frequency_1[letter[:-1]] = float(frequency)
    with open("Letter2_Freq.txt") as file:
        for line in file.readlines():
            try:
                frequency, letter = line.split("\t")
                letter_frequency_2[letter[:-1]] = float(frequency)
            except:
                continue


def get_fitness_array(file_content, matrix):
    """
    :param file_content: content of encryption file
    :param matrix: the matrix which represents the whole population of different permutations
    :return:
    """
    fitness_array = [] # array of fitness of every permutation
    for i in range(population_size):
        # permutation is row of matrix
        permutation = matrix[i, :]
        # replace letters in encryption according to permutation
        encryption = replace_letters(file_content, permutation)
        # call fitness function to find fitness of permutation
        fitness_i = find_fitness(encryption)
        fitness_tuple = (i, fitness_i) # tuple with fitness of permutation and its row in the matrix
        fitness_array.append(fitness_tuple)
    # sort fitness array according to fitness
    sorted_fitness_array = sorted(fitness_array, key=lambda x: x[1])

    return sorted_fitness_array


# receives 120 permutations and return 60 only.
def crossover(permutations):
    """
    selects random pair of permutation and does crossover between them
    :param permutations: list of 120 permutations
    :return: crossover results - list of 60 permutations
    """
    crossover_results = []
    num = len(permutations) - 1
    copied_permutations = permutations.copy()  # Create a copy of the permutations list

    for i in range(int(len(permutations) / 2)):
        per_one_idx = per_two_idx = 0
        # while the 2 random indexes are equal, choose 2 random indexes again
        while per_one_idx == per_two_idx:
            per_one_idx = random.randint(0, num)
            per_two_idx = random.randint(0, num)

        one = copied_permutations[per_one_idx]  # Use the copied_permutations list
        two = copied_permutations[per_two_idx]  # Use the copied_permutations list

        # pop indexes from permutations list so that the algorithm does not choose them again
        if per_one_idx > per_two_idx:
            copied_permutations.pop(per_one_idx)
            copied_permutations.pop(per_two_idx)
        else:
            copied_permutations.pop(per_two_idx)
            copied_permutations.pop(per_one_idx)
            
        num -= 2
        index = random.randint(1, permutation_size - 2)  # choose random index for crossover
        new_permutation = []

        new_permutation[0:index] = one[0:index] # the first part of new permutation is from the first part of the first permutation
        new_permutation[index:] = two[index:]# the second part of new permutation is from the second part of the second permutation
        new_permutation = check_permutation(new_permutation) # check new permutation is legal
        crossover_results.append(new_permutation)

    return crossover_results


def check_permutation(permutation):
    """
    checks the permutation is legal(contains every letter in ABC's exactly once) and if it is not, change it so that it is legal
    :param permutation: new permutation created by crossover
    :return:
    """
    missing = list(set(letters) - set(permutation)) # list of missing letters from permutation
    seen = [] # array of letter already encountered in permutation
    idx = 0
    for letter in permutation:
        if letter in seen:
            # if letter was already encountered in permutation, replace it with random missing letter
            choice = random.choice(missing)
            permutation[idx] = choice
            missing.remove(choice)
        else:
            seen.append(letter)
        idx += 1
    return permutation


def optimization_permutation(file_content, permutation):
    """
    This is the function of the optimization we used in Darwin's and Lamark's algorithms
    It creates a random mutation and only applies it to our permutation if it improves its fitness
    :param file_content: content from encryption file
    :param permutation: possible solution
    :return:
    """
    encryption = replace_letters(file_content, permutation) # replace letters according to encryption
    temp_permutation = copy.deepcopy(permutation) # create temporary copy of permutation
    fitness_current = find_fitness(encryption) # find current fitness of encryption

    index_1 = index_2 = 0
    # chooses 2 different random indexes
    while index_1 == index_2:
        index_1 = random.randint(0, permutation_size - 1)
        index_2 = random.randint(0, permutation_size - 1)

    # Switch the values at the selected indices to create mutation
    temp_permutation[index_1], temp_permutation[index_2] = temp_permutation[index_2], temp_permutation[index_1]
    encryption = replace_letters(file_content, temp_permutation) # replace letters in encryption according to permutation
    fitness_after = find_fitness(encryption) # fitness after the switching of the 2 indexes (mutation)

    # if fitness is improved by the mutation, change permutation to the mutation
    if fitness_after < fitness_current:
        permutation = copy.deepcopy(temp_permutation)

    return permutation


def optimization_matrix(file_content, matrix):
    """
    applies optimization to every row/permutation in the matrix
    :param file_content: content of encryption file
    :param matrix:
    :return:
    """
    matrix_opt = copy.deepcopy(matrix)
    for i in range(population_size):
        matrix_opt[i, :] = optimization_permutation(file_content, matrix[i, :])
    return matrix_opt


def mutations(permutations, prob):
    """
    :param permutations:
    :param prob: probability of mutation
    :return:
    """
    new_population = []
    # for the best 5*k in the population, don't do mutation - pass them as is to next generation
    for permutation in permutations[:5*k]:
        new_population.append(permutation)

    # for the rest of the population, do mutations
    for permutation in permutations[5*k:]:
        # create mutation according to probability
        if random.uniform(0, 1) <= prob:
            # choose 2 random indexes
            index_1 = random.randint(0, permutation_size - 1)
            index_2 = random.randint(0, permutation_size - 1)

            # Switch the values at the selected indices
            permutation[index_1], permutation[index_2] = permutation[index_2], permutation[index_1]

        new_population.append(permutation)
    return new_population

def calc_average_fitness(fitness_array):
    """
    :param fitness_array: array of fitness of every solution
    :return: the average fitness of the whole matrix/population
    """
    fitness_sum=0
    for entry in fitness_array:
        fitness=entry[1]
        fitness_sum+=fitness

    average_fitness=fitness_sum/population_size
    return average_fitness


def find_next_generation(file_content, matrix, algorithm_type):
    """
    :param file_content: the content of encryption file
    :param matrix: population of permutations
    :param algorithm_type: lamark, darwin of regular
    :return: new_matrix, lowest_fitness, lowest_fitness_prem,fitness_array
    """
    next_generation = [] # list of all permutations we pass to next generation
    crossover_population = [] # list of all permutations we pass to crossover function.
    fitness_array = [] # list of fitness scores to all permutations

    if algorithm_type == regular:
        fitness_array = get_fitness_array(file_content, matrix)

    if algorithm_type == darwin:
        # in darwin's algorithm, we do optimizations to all of the population,
        # calculate the fitness on the optimize population,
        # and pass the *original* permutation to the "cross-over",
        # to the "mutations" and to the next generation
        matrix_opt = optimization_matrix(file_content, matrix)
        fitness_array = get_fitness_array(file_content, matrix_opt)

    if algorithm_type == lamark:
        # in lamark's algorithm, we do optimizations to all of the population,
        # calculate the fitness on the optimize population,
        # and pass *them* to the "cross-over",
        # to the "mutations" and to the next generation
        matrix = optimization_matrix(file_content, matrix)
        fitness_array = get_fitness_array(file_content, matrix)

    lowest_fitness_prem = matrix[fitness_array[0][0]] # save lowest fitness permutation
    if generation % 10 == 0: # display the lowest fitness permutation every 10 generations
        print("lowest_fitness prem: " + str(lowest_fitness_prem))
    lowest_fitness = fitness_array[0][1] # save lowest fitness value

    for i in range(5*k):
        # the best 5*k permutations are passed as is to next generation and also copied and inserted 8 times  each into the crossover population
        row_idx = fitness_array[i][0] # save row of the ith best permutation from fitness array which contains tuples of (row number,fitness score) of permutations
        permutation = matrix[row_idx, :]  # save the the ith best permutation from matrix
        next_generation.append(permutation) # add permutation to next generation
        for j in range(8): # do 8 times
            permutation = copy.deepcopy(permutation)
            crossover_population.append(permutation) # add permutation to crossover list
    for i in range(5*k, 20*k):
        # next 15*k best permutations are passed as is to next generation and also copied and inserted 4 times  each into the crossover population
        row_idx = fitness_array[i][0]
        permutation = matrix[row_idx, :]
        next_generation.append(permutation)
        for j in range(4): # do 4 times
            permutation = copy.deepcopy(permutation)
            crossover_population.append(permutation)
    for i in range(20*k, 25*k):
        # the next 5*k best permutations are passed as is to next generation and to the crossover population
        row_idx = fitness_array[i][0]
        permutation = matrix[row_idx, :]
        next_generation.append(permutation)
        copied_permutation = copy.deepcopy(permutation)
        crossover_population.append(copied_permutation)
    for i in range(15*k):
        # next we create 15*k random permutation and pass them to next generation and crossover population in order to create more variety
        letters_set = set(letters)
        letters_list = list(letters_set)
        random_permutation = random.sample(letters_list, len(letters_list)) # generate random permutation
        next_generation.append(random_permutation)
        random_copied_permutation = copy.deepcopy(random_permutation)
        crossover_population.append(random_copied_permutation)

    next_generation += crossover(crossover_population) # add crossover function output to next generation list

    next_generation = mutations(next_generation, 0.5) # pass next generation list to mutations function and save result in next generation

    #create new matrix and add to it the whole next generation population. matrix is returned by function
    new_matrix = np.empty((population_size, permutation_size), dtype='U1')
    for i in range(population_size):
        new_matrix[i, :] = next_generation[i]

    return new_matrix, lowest_fitness, lowest_fitness_prem,fitness_array

def find_accuracy(permutation):
    """
    finds accuracy by comparing our current permutation to the correct encryption for this excercise.
    only works here because we know what the correct permutation is
    :param permutation:
    :return:
    """
    accuracy=0
    correct_perm=['Y', 'X', 'I', 'N', 'T', 'O', 'Z', 'J', 'C', 'E', 'B', 'L', 'D', 'U', 'K', 'M', 'S', 'V', 'P', 'Q', 'R', 'H', 'W', 'G', 'A', 'F']
    for i in range(permutation_size):
        if permutation[i]==correct_perm[i]:
            accuracy+=1
    accuracy/=permutation_size
    accuracy*=100

    return accuracy


def main_simulation_fitness_accuracy():
    """
    we ran this simulation of automation of main 10 times for each algorithm type to find lowest fitness and accuracy for each generation
    we then wrote the average fitness in each generation in each iteration of algorithm to an excel file for each algorithm type
    we used these excel files to plot the graphs.
    """
    global generation
    for algorithm_type in [regular, lamark, darwin]:
        for i in range(10):
            matrix = initialize()
            file_content = read_file()
            read_dictionary()
            read_letter_frequency()
            lowest_fitness = 1000
            counter = 0
            lowest_fitness_prem = []
            generation=0
            accuracy_per_iteration = []
            fitness_per_iteration = []

            while generation < 200 and counter <= 40:
                matrix, current_lowest_fitness, lowest_fitness_prem,fitness_array = find_next_generation(file_content, matrix, algorithm_type)

                if current_lowest_fitness != lowest_fitness:
                    lowest_fitness = current_lowest_fitness
                    counter = 0
                else:
                    counter += 1
                print("Type: ", algorithm_type, " ",i)
                print("Generation: ", generation)
                print("Lowest Fitness: ", lowest_fitness)
                generation += 1
                accuracy=find_accuracy(lowest_fitness_prem)
                accuracy_per_iteration.append(accuracy)
                fitness_per_iteration.append(current_lowest_fitness)
            # add the highest accuracy in each generation to csv file
            with open(algorithm_type + "_accuracy.csv","a") as output:
                row=""
                for accuracy in accuracy_per_iteration:
                    row+=str(accuracy)+","
                for i in range(200-generation):
                    row+=str(accuracy)+","
                row+="\n"
                output.write(row)
            # add the lowest fitness in each generation to csv file
            with open(algorithm_type + "_fitness.csv","a") as output:
                row=""
                for fitness in fitness_per_iteration:
                    row+=str(fitness)+","
                for i in range(200 - generation):
                    row += str(fitness) + ","
                row+="\n"
                output.write(row)

            print("\n\n" + str(algorithm_type) + " algorithm found the solution in "
                  + str(generation - 1) + " steps (the solution first discover in generation "
                  + str(generation - 1 - counter) + "). \nIt used a population size of " + str(population_size) + ".")

def main_simulation_fitness_average():
    """
    we ran this simulation of automation of main 10 times for each algorithm type to find average fitness for each generation
    we then wrote the average fitness in each generation in each iteration of algorithm to an excel file for each algorithm type
    we used these excel files to plot the graphs.
    :return:
    """
    global generation
    for algorithm_type in [regular,lamark,darwin]:
        for i in range(10):
            matrix = initialize()
            file_content = read_file()
            read_dictionary()
            read_letter_frequency()
            lowest_fitness = 1000
            counter = 0
            lowest_fitness_prem = []
            generation=0
            accuracy_per_iteration = []
            fitness_per_iteration = []
            avg_fitness_per_iteration=[]

            while generation < 200 and counter <= 40:
                matrix, current_lowest_fitness, lowest_fitness_prem, fitness_array= find_next_generation(file_content, matrix, algorithm_type)

                if current_lowest_fitness != lowest_fitness:
                    lowest_fitness = current_lowest_fitness
                    counter = 0
                else:
                    counter += 1

                print("Type: ", algorithm_type, " ",i)
                print("Generation: ", generation)
                print("Lowest Fitness: ", lowest_fitness)
                average_fitness = calc_average_fitness(fitness_array)
                print("Avearge Fitness: ",average_fitness)
                generation += 1
                accuracy=find_accuracy(lowest_fitness_prem)
                accuracy_per_iteration.append(accuracy)
                fitness_per_iteration.append(current_lowest_fitness)
                avg_fitness_per_iteration.append(average_fitness)
            # add the average fitness in each generation to csv file
            with open(algorithm_type + "_average_fitness.csv","a") as output:
                row=""
                for avg_fitness in avg_fitness_per_iteration:
                    row+=str(avg_fitness)+","
                for i in range(200 - generation):
                    row += str(avg_fitness) + ","
                row+="\n"
                output.write(row)

            print("\n\n" + str(algorithm_type) + " algorithm found the solution in "
                  + str(generation - 1) + " steps (the solution first discover in generation "
                  + str(generation - 1 - counter) + "). \nIt used a population size of " + str(population_size) + ".")




def main():
    """
    this is the regular main function of our program.
    It calls next generation function in a loop until algorithm convergence or generation 200.
    """
    global generation
    print("Loading...")
    algorithm_type_input = ""
    tries=0
    while algorithm_type_input not in ["Lamark","Darwin", "Regular"]: # continues while algorithm type in not valid
        if tries>0:
            print("Error: please enter valid algorithm type.") # print error message if user did not enter valid algorithm type
        algorithm_type_input=input("Please enter algorithm type (Lamark, Darwin or Regular):") # user inputs algorithm type
        tries+=1
    # check the user input and set algorithm type according to it
    if algorithm_type_input=="Lamark":
        algorithm_type=lamark
    if algorithm_type_input=="Darwin":
        algorithm_type=darwin
    if algorithm_type_input=="Regular":
        algorithm_type=regular

    matrix = initialize() # initialize matrix
    file_content = read_file() # reads file content
    read_dictionary() # reads dictionary
    read_letter_frequency() # reads letter frequency
    lowest_fitness = 1000
    counter = 0
    lowest_fitness_prem = []

    # the options of algorithm_type are: regular, darwin or lamark.
    # algorithm_type = darwin # the regular here can be switched to lamark of darwin if we want to run their algorithm

    # continues until generation 200 or until lowest fitness doesn't change for 40 generations -- convergence
    while generation < 200 and counter <= 40:
        # call next generation function to find next generation
        matrix, current_lowest_fitness, lowest_fitness_prem,fitness_array = find_next_generation(file_content, matrix, algorithm_type)
        # count the number of times the lowest fitness stays the same. used to find algorithm convergence
        if current_lowest_fitness != lowest_fitness:
            lowest_fitness = current_lowest_fitness
            counter = 0
        else:
            counter += 1
        # display generation and lowest fitness
        print("Algorithm Type: ", algorithm_type)
        print("Generation: ", generation)
        print("Lowest Fitness: ", lowest_fitness)
        generation += 1

    real_text = replace_letters(file_content, lowest_fitness_prem) # finds real text by replacing letters in encryption according the best permutation which was found by algorithm
    with open("plain.txt", "w") as file: # adds real text to plain.txt file
        file.write(real_text)
    with open("perm.text", "w") as out_file: # adds real permutation to perm.txt file
        for letter in letters:
            row = letter + " " + lowest_fitness_prem[letters_idx_map[letter]] + "\n"
            out_file.write(row)

    # print number of generations to find solution and  the population size and the total number of calls to fitness function
    print("\n\n" + str(algorithm_type) + " algorithm found the solution in "
          + str(generation-1) + " generations (the solution first discover in generation "
          + str(generation-1-counter) + "). \nIt used a population size of " + str(population_size) + "."
          + "\nThe total number of calls to fitness function: " + str(total_fitness_calls)+ ".")
    input("Press Enter to exit...")


if __name__ == '__main__':

    main()