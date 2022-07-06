#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 17 18:15:31 2022

@author: auro
"""
import random
import numpy as np
from deap import base
from deap import creator
from deap import tools
from deap import algorithms
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from pprint import pprint

#Clase GA_melody

class GA_melody:

    # Class variables
    
    samplerate = 44100 #Frequecy in Hz
    """The sampling frequency or sampling rate, fs, is the average number of samples
     obtained in one second, thus fs = 1/T. Its units are samples per second or 
     hertz e.g. 48 kHz is 48,000 samples per second."""
     
    amplitude = 4096*1.5

    ref_notes_dict = {'C': 261.626,
                      'c': 277.183091524845,
                      'D': 293.665255850988,
                      'd': 311.1275006697019,
                      'E': 329.6281045997961,
                      'F': 349.2288116870894,
                      'f': 369.9950374694234,
                      'G': 391.99608729493866,
                      'g': 415.3053876222321,
                      'A': 440.00073107433656,
                      'a': 466.16453606436875,
                      'B': 493.8841218593214}

    modes_dict = {"major": [2, 2, 1, 2, 2, 2, 1],
                  "minor": [2, 1, 2, 2, 1, 2, 2],
                  "dorian": [2, 1, 2, 2, 2, 1, 2],
                  "pentatonic" : [3,2,2,3,2],
                  "blues" : [3,2,1,1,3,2]}
    
    tonality_chords = {'major': {'1': 'major',
                                 '2': 'minor',
                                 '3': 'minor',
                                 '4': 'major',
                                 '5': 'major',
                                 '6': 'minor',
                                 '7': 'dim'},
                       'minor': {'1': 'minor',
                                 '2': 'dim',
                                 '3': 'major',
                                 '4': 'minor',
                                 '5': 'minor',
                                 '6': 'major',
                                 '7': 'major'} }
    
        
    correspondence = {"major" : {'1': 1,
                                 '2': 3,
                                 '3': 5,
                                 '4': 6,
                                 '5': 8,
                                 '6': 10,
                                 '7': 12} ,
                      "minor" : {'1': 1,
                                 '2': 3,
                                 '3': 4,
                                 '4': 6,
                                 '5': 8,
                                 '6': 9,
                                 '7': 11} }                          
    
    type_of_chords_dict = {"major": [1,5,8],
                      "aum": [1,5,9],
                      "minor": [1,4,8],
                      "dim": [1,4,7]}
    
    # --------------------------------------------------------------------------

    def __init__(self, 
                 tonality, 
                 mode, 
                 range_of_tones, 
                 pulses, 
                 shortest_figure, 
                 shortest_figure_dur, 
                 melody_structure,
                 rpr,
                 hpr,
                 chord_progression):
        self.tonality = tonality
        self.mode = mode
        self.range_of_tones = range_of_tones  # Example of attribute of a class
        self.pulses = pulses
        self.shortest_figure = shortest_figure
        self.shortest_figure_dur = shortest_figure_dur
        self.melody_structure = melody_structure
        self.rpr = rpr
        self.hpr = hpr
        self.chord_progression = chord_progression
        self.notes_wo_silence_and_repetition = GA_melody.get_tonality_notes(
            self)
        self.notes_freq = GA_melody.get_tonality_notes_out(self)
        self.notes = [0] + self.notes_wo_silence_and_repetition + [-1]
        self.num_of_bars = self.pulses/4 #Solo para 4/4
        self.pulses_per_bar = 4
        self.max_notes_per_bar = int(self.num_of_bars * self.shortest_figure)
        self.bar_duration = self.shortest_figure * self.shortest_figure_dur
        self.overall_dur = self.num_of_bars * self.bar_duration

        
        #Random melody
        if self.melody_structure == []:
            melody_vec = [self.notes[random.randint(0, len(self.notes)-2)]] #Para no empezar con repetición
            for i in range(1, self.pulses):
                rand_note = random.randint(0, len(self.notes)-1)
                melody_vec.append(self.notes[rand_note])
        else:
                #Random melody from structure
                melody_vec = self.melody_structure
                for i in range(0, len(melody_vec)):
                    if melody_vec[i] == -3:
                        rand_note = random.randint(0, len(self.notes)-1)
                        melody_vec[i] = self.notes[rand_note]
        self.melody = melody_vec
        
        self.fitness = GA_melody.fitness_test(self)
    
    def random_note(self):
        rand_note = random.randint(0, len(self.notes)-1)
        return self.notes[rand_note]
        
    def __len__(self):
        return len(self.melody)
    
    def __add__(self,other):
        return self.melody + other.melody

    def get_tonality_notes(self):
        mode = (GA_melody.modes_dict[self.mode]*self.range_of_tones)[0:self.range_of_tones-1] #MEJORABLE, MULTIPLICO DE MÁS
        i = 1
        good_notes = [i]
        for j in mode:
            i = i+j
            good_notes.append(i)
        return good_notes
    
    def get_tonality_notes_out(self):
        ref = GA_melody.ref_notes_dict[self.tonality]
        mode = (GA_melody.modes_dict[self.mode]*self.range_of_tones)[0:self.range_of_tones-1]
        i = 0
        good_notes = [i]
        for j in mode:
            i = i+j
            good_notes.append(i)
        notes = [ref * pow(2,(i/12)) for i in good_notes]
        return notes

    def random_melody(self):
        notes = GA_melody.get_tonality_notes(self)
        random.seed(10)  # Preguntar
        melody_vec = [notes[random.randint(0, len(notes)-2)]]
        for i in range(1, self.pulses):
            rand_note = random.randint(0, len(notes)-1)
            melody_vec.append(notes[rand_note])
        return melody_vec
    
    def fitness_test(self):    
        # Unison, perfect fourth, perfect 5th, octave (1 point)
        perfect_consonants = [0, 5, 7, 12]
        # Minor and major thirds and sixths (2 points)
        imperfect_consonants = [3, 4, 8, 9]
        # Seconds(3 points)
        seconds = [1, 2]
        # Sevenths (3 points)
        sevenths = [10, 11]

        interval_vec = []
        i = 0
        j = i+1
        fitness_points = 0
        while i < (len(self.melody)-1) and j<(len(self.melody)):
            if self.melody[j] != (-1) and self.melody[j] !=0:
                interval_vec.append(abs(self.melody[j]-self.melody[i]))
                i += 1
                j = i+1
            else:
                #Silencio o repetición
                if self.melody[j] == (0):
                    fitness_points += 2
                if self.melody[j] == (-1):
                    fitness_points -= 30
                j += 1
                
        for p in interval_vec:
            if p in perfect_consonants:
                fitness_points -= 10
            elif p in imperfect_consonants:
                fitness_points -= 5
            elif p in seconds or p in sevenths:
                fitness_points += 0
            else:
                fitness_points += 0

        return fitness_points
    
    @staticmethod
    def straight_envelope(sa=1, at=0.3, dt=0.5, dur = 1):
        at_int = int(at*GA_melody.samplerate)
        dt_int = int(dt*GA_melody.samplerate)
        return np.concatenate([(sa/at_int)*np.arange(0,at_int+1),
                              sa*np.ones(int(GA_melody.samplerate*dur - (at_int+1+dt_int+1))),
                              (-sa/dt_int)*np.arange(-dt_int,1)])   
    
    @staticmethod
    def get_wave_env(freq, duration, sa = 1, at = 0.01, dt = 0.01):
        t = np.linspace(0., duration, int(GA_melody.samplerate*duration))
        wave = GA_melody.amplitude*np.sin(2*np.pi*freq*t)
        env = GA_melody.straight_envelope(sa, at, dt, dur = duration)
        return wave*env
    
    @staticmethod
    def chord_maker(freq, dur, type_of_chord):
         return sum([GA_melody.get_wave_env(freq*pow(2,(i-1)/12), dur) for i in GA_melody.type_of_chords_dict[type_of_chord]])
     
    def chord_progression(self):
        return np.concatenate([GA_melody.chord_maker(self.notes_freq[int(chord)-1],
                                           self.bar_duration,
                                           GA_melody.tonality_chords[self.mode][chord]) 
                           for chord in self.chord_progression])

    def melody_plus_chords(self):
        return np.array([GA_melody.get_song_data_all_notes(self), GA_melody.chord_progression(self)]).T
        #return GA_melody.get_song_data(self) + GA_melody.chord_progression(self)
        
    def melody_plus_chords_sum(self):
        suma = GA_melody.get_song_data_all_notes(self) + GA_melody.chord_progression(self)
        data = suma * (16300/np.max(suma)) # Adjusting the amplitude 
        return data 
    
    def get_song_data_all_notes(self, title = ""):
        '''
        Function to concatenate all the waves (notes)
        '''
        
        long = self.shortest_figure_dur
        ref = GA_melody.ref_notes_dict[self.tonality]
        good_notes = [i for i in range(0, max(self.melody)+2)]
        note_freqs = [ref * pow(2,(i/12)) for i in good_notes[:-1]]
        dict_notes = dict(zip(good_notes[1:], note_freqs))
        song_tuples = []
        i = 0
        while i<len(self.melody):
            note = self.melody[i]
            dur = long
            if note != 0 and note!= -1:
                note = dict_notes[note]
        
            try:
                while self.melody[i+1] == -1:
                    dur += long
                    i += 1
            except:
                pass
            i += 1
            song_tuples.append((note,dur))
            
        #pprint(song_tuples)
        
        song = [GA_melody.get_wave_env(tuple_[0], tuple_[1]) for tuple_ in song_tuples]
        
        song = np.concatenate(song)
        4
        data = song * (16300/np.max(song)) # Adjusting the  (Optional)

        write(title+'_test_class.mp3', GA_melody.samplerate, data.astype(np.int16))
        
        return data
    
    @staticmethod
    #MEJORAR
    def mel_out(melody):
        mel = GA_melody("C", "major", 
                          range_of_tones = 10, 
                          pulses = 16, 
                          shortest_figure = 8, 
                          shortest_figure_dur = 0.4, #Revisar para shortest_duration = 0.35
                          melody_structure =  [], 
                          rpr = 0.2,
                          hpr = 0.4,
                          chord_progression = "1451") 
        mel.melody = melody
        mel_plus_chords = GA_melody.melody_plus_chords_sum(mel)
        write('out3.mp3', GA_melody.samplerate, mel_plus_chords.astype(np.int16))

def interval_vector(melody):
    interval_vec = []
    i = 0
    j = i+1
    while i < (len(melody)-1) and j<(len(melody)):
        if melody[i] == (-1) or melody[i] == 0:
               i += 1
               j = i+1
        elif melody[j] != (-1) and melody[j] !=0:
            interval_vec.append((melody[j]-melody[i]))
            i += 1
            j = i+1
        else:
            j += 1
    return interval_vec

def f1(x,GA_melody_instance):
    """
    Checks whether a note belongs to the chord playing in the same beat.

    Parameters
    ----------
    x : list of ints
        Melody
    GA_melody_instance : GA melody instance
        DESCRIPTION. The default is mel.

    Returns
    -------
    normalization : float
        Number of notes that belong to their correspponding chord divided
        by the total amount of notes.

    """
    chord_progression = GA_melody_instance.chord_progression
    note_in_chord_counter = 0
    notes_per_bar_counter = 0
    notes_per_bar = mel.shortest_figure
    mode = mel.mode
    for chord in chord_progression:
        root_note = GA_melody.correspondence[mode][chord]
        type_of_chord = GA_melody.tonality_chords[mode][chord]
        third = (root_note + (GA_melody.type_of_chords_dict[type_of_chord][1]-1))
        fifth = (root_note + (GA_melody.type_of_chords_dict[type_of_chord][2]-1))
        """
        print(root_note, third, fifth)
        print(notes_per_bar_counter, notes_per_bar_counter + notes_per_bar)
        print(x[notes_per_bar_counter: notes_per_bar_counter + notes_per_bar])
        """
        for note in x[notes_per_bar_counter: notes_per_bar_counter + notes_per_bar]:
            if note != 0 and note != (-1):
                if note%12 in (root_note, third, fifth):
                    note_in_chord_counter += 1
        notes_per_bar_counter += notes_per_bar
    
    normalization = note_in_chord_counter/len(x)
    return normalization
    
def f2(interval_vec_x):
    interval_vec = interval_vec_x
    if len(interval_vec) < 1:
        return 0
    num_of_intervals = len(interval_vec)
    one_step_counter = 0
    two_step_counter = 0
    same_note_counter = 0
    perfect_fourth_counter = 0
    perfect_fifth_counter = 0
    for i in interval_vec:
        """
        Checks whether next note's scale degree is one tone
        higher or lower than the previous one
        """
        if i == 2 or i == -2: 
            one_step_counter += 1
            
        """
        Checks whether next note's scale degree is two tones
        higher or lower than the previous one
        """
        if i == 4 or i == -4: 
            two_step_counter += 1
        
        """
        Checks whether next note's scale degree is the same 
        note than the previous one
        """
        if i == 0: 
            same_note_counter += 1
            
        """
        Checks whether next note's scale degree is
        at a Perfect Fourth distance from the previous one
        """
        if i == 5 or i == -5: 
            perfect_fourth_counter += 1
        
        """
        Checks whether next note's scale degree is
        at a Perfect Fifth distance from the previous one
        """
        if i == 7 or i == -7: 
            perfect_fifth_counter += 1
        
    weights = np.array([1,1,0.9,0.8,0.7])
    normalized_counters = (1/num_of_intervals) * np.array([one_step_counter,
                                                       two_step_counter,
                                                       same_note_counter,
                                                       perfect_fourth_counter,
                                                       perfect_fifth_counter])
    return np.dot(weights, normalized_counters)

def f3(interval_vec_x):
    interval_vec = interval_vec_x
    if len(interval_vec) <= 1:
        return 0
    num_of_intervals = len(interval_vec)
    raising_melody_counter = 0
    falling_melody_counter = 0
    stable_melody_counter = 0
    i = 0
    while i < (num_of_intervals-1):
        
        """
        Checks whether three consecutive notes
        form a raising melody
        """
        if interval_vec[i]>0 and interval_vec[i+1]>0:
            raising_melody_counter += 1
        
        """
        Checks whether three consecutive notes
        form a falling melody
        """
        if interval_vec[i]<0 and interval_vec[i+1]<0:
            falling_melody_counter += 1
        
        """
        Checks whether three consecutive notes
        form a stable melody
        """
        if interval_vec[i]==0 and interval_vec[i+1]==0:
            stable_melody_counter += 1
        
        i += 1
    
    weights_3 = np.array([1,1,0.9])
    normalized_counters = (1/(num_of_intervals-1)) * np.array([raising_melody_counter,
                                                             falling_melody_counter,
                                                             stable_melody_counter])
    return np.dot(weights_3, normalized_counters)

def f4(x):
    """
    Checks whether a melody starts with the tonality's root note
    """
    if x[0]%12 == 1:
        return 1.
    else:
        return 0.
    
def f5(x):
    """
    Checks whether a melody ends with the tonality's root note
    """
    i = -1
    while (x[i] == 0 or x[i] == (-1) and i>0):
        i -= 1
    if x[i]%12 == 1:
        return 1.
    else:
        return 0.

def f6(interval_vec_x):
    interval_vec = interval_vec_x
    if len(interval_vec) <= 1:
        return 0
    over_fifth_violation_counter = 0
    for i in interval_vec:
          
        """
        Checks whether next note's scale degree is
        at a higher distance that a Perfect Fifth
        """
        if abs(i)>8:
            over_fifth_violation_counter += 1
    
    normalization_over_fifth = over_fifth_violation_counter/len(interval_vec)
    return normalization_over_fifth
    
def f7(x):
    notes_dur = []
    long = 1
    i = 0
    while i<len(x):
        dur = long    
        try:
            while x[i+1] == -1:
                dur += long
                i += 1
        except:
            pass
        i += 1
        notes_dur.append(dur)
    
    """
    Checks wheteher there's a drastic duration change
    between two consecutive notes and if affirmative, penalizes it
    """
    over_duration_violation_counter = 0
    j=0
    while j<(len(notes_dur)-1):
        if notes_dur[j+1]/notes_dur[j]>4 or notes_dur[j]/notes_dur[j+1]>4:
            over_duration_violation_counter += 1
        j += 1
    
    normalization_over_duration = over_duration_violation_counter/len(notes_dur)
    return normalization_over_duration

def f8(x, rpr):
    """
    Checks the ratio of total rest durations 
    to the overall duration.

    Parameters
    ----------
    x : TYPE, list.
        Melody.
    rpr : TYPE, float between 0 and 1.
        The default is 0.2.

    Returns
    -------
    A float in [0,1].

    """
    rest_counter = 0
    for i in x:
        if i == 0:
            rest_counter += 1
    rest_prop = rest_counter/len(x)
    
    if abs(rest_prop-rpr)<=0.05:
        return -200*(rest_prop-rpr)**2+1
    else:
        return 0
        
def f9(x, hpr):
    """
    Checks the ratio of total hold events
    to the overall duration.

    Parameters
    ----------
    x : TYPE, list.
        Melody.
    rpr : TYPE, float between 0 and 1.
        The default is 0.4.

    Returns
    -------
    A float in [0,1).

    """
    hold_counter = 0
    for i in x:
        if i == 0:
            hold_counter += 1
    hold_prop = hold_counter/len(x)
    
    if abs(hold_prop-hpr)<=0.1:
        return -50*(hold_prop-hpr)**2+1
    else:
        return 0

def funcion_objetivo(melody, rpr, hpr, x):
    """
    Función objetivo de nuestro problema
    """
    interval_vec = interval_vector(x)
    
    #Chord note
    f_1 =f1(x, melody)
    
    #f2 Relationship between notes
    f_2 = f2(interval_vec)
    
    #f3 Direction
    f_3 = f3(interval_vec)
    
    #f4 Beginning note
    f_4 = f4(x)
    
    #f5 Ending note
    f_5 = f5(x)
        
    #f6 Over fifth
    f_6 = f6(interval_vec)
    
    #f7 Drastic duration changes
    f_7 = f7(x)
    
    #f8 Rest proportion
    f_8 = f8(x, rpr)
    
    #f9 Hold event proportion
    f_9 = f9(x, hpr)
    
    #f10 Pattern matching
    
    weights_f = (1./7)*np.ones(7)
    f_values = np.array([f_1, f_2, f_3, f_4, f_5, f_8, f_9])
    
    #f stands for fitness
    f = np.dot(weights_f, f_values)
    f -= f_6
    f -= f_7
    
    return f,

def cxboth(a,b):
    random_number = random.random()
    if random_number<=0.5:
        c = tools.cxOnePoint(a,b)
        return c
    else:
        c = tools.cxTwoPoint(a,b)
        return c
    
def mutation_1(prob, a):
    random_number = random.random()
    if random_number < prob:
        # Change octave
        i = random.randint(0, len(a)-1)
        sign = random.choice([-1, 1])
        if (a[i] + 12*sign) > 0 and a[i]!=0 and a[i]!=(-1):
            a[i] = a[i] + 12*sign
    return(a,)

def mutation_2(prob,a):
    random_number = random.random()
    if random_number < prob:
        # Change random note
        index = random.randint(1, len(a)-1)
        rand_note = random.randint(-1, max(a))
        a[index] = rand_note
    return(a,)

def mutation_3(prob, a):
    random_number = random.random()
    if random_number < prob:
        # Swapping two consecutive notes
        i = random.randint(0, len(a)-2)
        a[i], a[i+1] = a[i+1], a[i]
    return(a,)


def mutation_4(prob, a):
    random_number = random.random()
    if random_number < prob:
        #Repeating the last note
        index = random.randint(1, len(a)-1)
        a[index] = a[index-1]
    return(a,)

def mutation_5(prob, a):
    #Making a note longer
    c = tools.mutUniformInt(a, low=-1, up=-1, indpb = prob)
    return c

def mutation_6(prob, a):
    #Change note for silence
    c = tools.mutUniformInt(a, low=0, up=0, indpb = prob)
    return c

def mut_all(probs_vect, distr_vect, a):
    random_number = random.random()
    if random_number<= distr_vect[0]:
        c = mutation_1(probs_vect[0], a)
        return c
    if random_number > distr_vect[0] and random_number <= distr_vect[1]:
        c = mutation_2(probs_vect[1], a)
        return c
    if random_number > distr_vect[1] and random_number <= distr_vect[2]:
        c = mutation_3(probs_vect[2], a)
        return c
    if random_number > distr_vect[2] and random_number <= distr_vect[3]:
        c = mutation_4(probs_vect[3], a)
        return c
    if random_number > distr_vect[3] and random_number <= distr_vect[4]:
        c = mutation_5(probs_vect[4], a)
        return c
    else:
        c = mutation_6(probs_vect[5], a)
        return c

def plot_evolucion(log):

    #Representa la evolución del mejor individuo en cada generación

    gen = log.select("gen")
    #fit_mins = log.select("min")
    fit_maxs = log.select("máximo")
    fit_ave = log.select("media")

    fig, ax1 = plt.subplots()
    
    #ax1.plot(gen, fit_mins, "darkturquoise")
    ax1.plot(gen, fit_maxs, "deepskyblue")
    ax1.plot(gen, fit_ave, "darkcyan")
    
    ax1.set_xlabel("Generación")
    ax1.set_ylabel("Fitness")
    #ax1.set_ylim([-10, 160])
    ax1.legend(["Máximo", "Media"], loc="lower center")
    plt.grid(True)
    plt.savefig("Funcionamiento.eps", dpi = 300)

def main(CXPB, MUTPB, NGEN, NPOP):
    #random.seed(75) 
    pop = toolbox.population(NPOP) 
    hof = tools.HallOfFame(1) 
    stats = tools.Statistics(lambda ind: ind.fitness.values) 
    stats.register("media", np.mean)
    stats.register("d. típica", np.std)
    #stats.register("Mínimo", np.min)
    stats.register("máximo", np.max)
    logbook = tools.Logbook()
    pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=CXPB, 
                                       mutpb=MUTPB, ngen=NGEN, stats=stats, 
                                       halloffame=hof, verbose=True)
    return hof, logbook


if __name__ == "__main__":  
    rpr_u = 0.2
    hpr_u = 0.4
    mel = GA_melody("C", "major", 
                      range_of_tones = 10, 
                      pulses = 16, 
                      shortest_figure = 8, 
                      shortest_figure_dur = 0.4, 
                      melody_structure =  [], 
                      rpr = rpr_u,
                      hpr = hpr_u,
                      chord_progression = "1451") 
    
    mel_u = mel
    cxpb, mutpb, npop, ngen, t_size = 0.7, 0.4, 30, 30, 4
    
    toolbox = base.Toolbox()
    
    # Creamos los objetos para definir el problema y el tipo de individuo
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
   
    # Generación de genes 
    toolbox.register("random_note_ga", GA_melody.random_note, mel)
   
    # Generación de inviduos y población
    toolbox.register("individual", tools.initRepeat, creator.Individual, 
                     toolbox.random_note_ga, mel.max_notes_per_bar)
    toolbox.register("population", tools.initRepeat, list, 
                     toolbox.individual)
    
    toolbox.register("evaluate", funcion_objetivo, mel_u, mel.rpr, mel.hpr)
    toolbox.register("select", tools.selTournament, tournsize = t_size)
    toolbox.register("mate", cxboth)
    toolbox.register("mutate", mut_all, mutpb*np.array([0.1,0.1,0.1,0.1,0.1,0.1]), np.linspace(0,1,7)[1:-1])  
   

    best, log = main(cxpb, mutpb, ngen, npop) 
    print("Mejor fitness: %f" %best[0].fitness.values)
    print("Mejor individuo %s" %best[0])
    plot_evolucion(log)

    mel.melody = best[0]
    data = GA_melody.get_song_data_all_notes(mel)
    write('melodia.mp3', GA_melody.samplerate, data.astype(np.int16))
    mel_plus_chords = GA_melody.melody_plus_chords_sum(mel)
    write("melodia_y_acordes.mp3", GA_melody.samplerate, mel_plus_chords.astype(np.int16)) 


