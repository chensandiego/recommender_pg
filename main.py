import csv
import os

from math import sqrt
import datetime
from collections import defaultdict
import heapq
from operator import  itemgetter

def load_reviews(path,**kwargs):
    options={
        'fieldnames':('userid','movieid','rating','timestamp'),
        'delimiter':'\t',
    }

    options.update(kwargs)
    parse_date=lambda r,k: datetime.datetime.fromtimestamp(float(r[k]))
    parse_int = lambda r,k: int(r[k])

    with open(path,'r',encoding="ISO-8859-1") as reviews:
        reader=csv.DictReader(reviews,**options)
        for row in reader:
            row['movieid'] = parse_int(row,'movieid')
            row['userid'] = parse_int(row,'userid')
            row['rating'] = parse_int(row,'rating')
            row['timestamp'] = parse_date(row,'timestamp')
            yield row

def relative_path(path):
    """
    return a path relative from this code file
    """
    dirname = os.path.dirname(os.path.realpath('__file__'))
    path = os.path.join(dirname,path)
    return os.path.normpath(path)


def load_movies(path,**kwargs):
    options={
    'fieldnames':('movieid','title','release','video','url'),
    'delimiter' :'|',
    'restkey': 'genre',
    }
    options.update(kwargs)
    parse_date=lambda r,k: datetime.datetime.strptime(r[k],'%d-%b-%Y') if r[k] else None
    parse_int = lambda r,k: int(r[k])

    with open(path,'r',encoding="ISO-8859-1") as movies:
        reader=csv.DictReader(movies,**options)
        for row in reader:
            row['movieid'] = parse_int(row,'movieid')
            row['release'] = parse_date(row,'release')
            row['video'] = parse_date(row,'video')
            yield row

class MovieLens(object):
    """
    data structure to build our recomend
    """
    def __init__(self,udata,uitem):
        """
        instantiate with a path to u.data and u.item
        """
        self.udata= udata
        self.uitem=uitem
        self.movies={}
        self.reviews=defaultdict(dict)
        self.load_dataset()



    def load_dataset(self):
        """
        loads two datasets into memory, indexed on id
        """
        for movie in load_movies(self.uitem):
            self.movies[movie['movieid']] = movie

        for review in load_reviews(self.udata):
            self.reviews[review['userid']][review['movieid']]=review


    def reviews_for_movie(self,movieid):
        """
        yields the reviews for a given movie
        """
        for review in self.reviews.values():
            if movieid in review:
                yield review[movieid]

    def average_reviews(self):
        """
        avg the star rating for all movies. yields a tuple of movieid,
        the avg rating,and num of reviews
        """

        for movieid in self.movies:
            reviews= list(r['rating'] for r in self.reviews_for_movie(movieid))
            average=sum(reviews)/float(len(reviews))
            yield (movieid,average,len(reviews))

    def bayesian_average(self,c=59,m=3):
        """
        reports the bayesian avg with parameter c and m
        """
        for movieid in self.movies:
            reviews= list(r['rating'] for r in self.reviews_for_movie(movieid))

            #only want reviews more than 300
            #if do not care
            #remove if condition
            #only want the rating is more or equal to 4
#            if len(reviews) >300:
            average=((c * m)+ sum(reviews)) / float(c + len(reviews))

#                if average>=4.0:
            yield (movieid,average,len(reviews))

    def top_rated(self,n=10):
        """
        yields the n top rated movies
        """
        return heapq.nlargest(n,self.bayesian_average(),key=itemgetter(1))

    def shared_preferences(self,criticA,criticB):
        """
        returns the intersect of ratings for two critics, A and B
        """
        if criticA not in self.reviews:
            raise KeyError("coundn't find critic '%s' in data" % criticA)
        if criticB not in self.reviews:
            raise KeyError("coundn't find critic '%s' in data" % criticB)


        moviesA = set(self.reviews[criticA].keys())
        moviesB = set(self.reviews[criticB].keys())

        #interset operator
        shared= moviesA & moviesB

        reviews = {}
        for movieid in shared:
            reviews[movieid]= (
                self.reviews[criticA][movieid]['rating'],
                self.reviews[criticB][movieid]['rating'],
            )
        return reviews

    def euclidean_distance(self,criticA,criticB,prefs='users'):
        """
        reports the  euclidean distance of two critics, A and B by
        performing a J-dimensional euclidean calculation of each of their
        preference vectors for the intersection of books the critics have rated
        """

        if prefs=='users':
            preferences = self.shared_preferences(criticA,criticB)
        elif prefs=='movies':
            preferences = self.shared_critics(criticA,criticB)
        else:
            raise Exception("No preferences of type '%s'."%prefs)

        #if they have no rankings in common, return 0
        if len(preferences) == 0: return 0

        #sum the squares of the diffs
        sum_of_squares=sum([pow(a-b,2) for a,b in preferences.values()])

        #return the inverse of the distance to give a higher score
        #to folks who are more similar add 1 to prevent division by zero errors
        return 1/(1+sqrt(sum_of_squares))



    def pearson_correlation(self,criticA,criticB,prefs='users'):
        """
        returns the pearson correlation of two critics, A and B by
        performing the PPMC calculation on the scatter plot of (a,b)
        ratings on the shared set of critqued titles.
        """

        #get the set of mutually rated items
        if prefs=='users':
            preferences=self.shared_preferences(criticA,criticB)
        elif prefs=='movies':
            preferences=self.shared_critics(criticA,criticB)
        else:
            raise Exception("no preferences of type '%s' "% prefs)

        #store the length to save traversals of the len computation
        #if they have no rankings in common, return 0
        length = len(preferences)
        if length ==0: return 0

        #loop through the preferences of each critic once and compute the
        #various summations that are required for our final  calculation
        sumA = sumB = sumSquareA = sumSquareB = sumProducts = 0

        for a,b in preferences.values():
            sumA += a
            sumB += b
            sumSquareA +=pow(a,2)
            sumSquareB +=pow(b,2)
            sumProducts += a*b

        #calculate pearson score
        numerator = (sumProducts*length) - (sumA*sumB)
        denominator = sqrt(((sumSquareA*length)-pow(sumA,2))*((sumSquareB*length)-pow(sumB,2)))

        #prevent division by zero
        if denominator ==0: return 0

        return abs(numerator/denominator)

def main():
    data=relative_path('u.data')
    item=relative_path('u.item')
    model = MovieLens(data,item)

    for mid,avg,num in model.top_rated(10):
        title=model.movies[mid]['title']

        print ("[%0.3f average rating (%i reviews)] %s" % (avg,num,title))

    print("euclidean distance computation")
    print(model.euclidean_distance(232,532))

    print("pearson correlation computation")
    print(model.pearson_correlation(232,532))



if __name__=='__main__':
    main()
