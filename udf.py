import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import requests
import seaborn as sns
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.metrics import silhouette_samples



class transform:
    def __init__(self):
        return
# list to string
    def ListToString(list):
        string_words = ' '.join(list)
        return string_words

# string to list
    def StringToList(string):
        listRes = list(string.split(" "))
        return listRes




StopWords = stopwords.words("english")
StopWords.extend(["u","from"])


class clean:
    def __init__(self):
        return


class scrab:
    def __init__(self):
        return

    def extract_text(url):
        headers = {
            'User-Agent': "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.6; rv:2.0.1) Gecko/20100101 Firefox/4.0.1"
        }
        resp = requests.get(url, headers=headers)
        s = BeautifulSoup(resp.text, "html.parser")
        title = s.title
        text = s.get_text(strip=True)
        return title, text

    def title(title):
        title = str(title)
        title = title[title.find("Obama -") + len("Obama -"):title.find("</title>")]
        title = title[:title.find("(transcript-audio-video)")]
        title = title[:title.find("(text-audio-video)")].strip()
        return title

    ## delete something like cd, pdf ...
    def allowed(speech):
        allowed = speech[speech.find("transcribed directly from audio]")
                         + len("transcribed directly from audio]"):speech.find(
            "Book/CDs by Michael E. Eidenmuller,")].strip()
        return allowed


class k_means:
    def __init__(self):
        return

    # Transforms a centroids dataframe into a dictionary to be used on a WordCloud.
    def centroidsDict(centroids, index):
        a = centroids.T[index].sort_values(ascending=False).reset_index().values
        centroid_dict = dict()

        for i in range(0, len(a)):
            centroid_dict.update({a[i, 0]: a[i, 1]})

        return centroid_dict
    
    def printAvg(avg_dict):
        for avg in sorted(avg_dict.keys(), reverse=True):
            print("Avg: {}\tK:{}".format(avg.round(4), avg_dict[avg]))

    def plotSilhouette(df, n_clusters, kmeans_labels, silhouette_avg):
        fig, ax1 = plt.subplots(1)
        fig.set_size_inches(8, 6)
        ax1.set_xlim([-0.2, 1])
        ax1.set_ylim([0, len(df) + (n_clusters + 1) * 10])

        ax1.axvline(x=silhouette_avg, color="red",
                    linestyle="--")  # The vertical line for average silhouette score of all the values
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])
        plt.title(("Silhouette analysis for K = %d" % n_clusters), fontsize=10, fontweight='bold')

        y_lower = 10
        sample_silhouette_values = silhouette_samples(df,
                                                      kmeans_labels)  # Compute the silhouette scores for each sample
        for i in range(n_clusters):
            ith_cluster_silhouette_values = sample_silhouette_values[kmeans_labels == i]
            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values, facecolor=color,
                              edgecolor=color, alpha=0.7)

            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i,
                     str(i))  # Label the silhouette plots with their cluster numbers at the middle
            y_lower = y_upper + 10  # Compute the new y_lower for next plot. 10 for the 0 samples
        plt.show()

class util:
    def __init__(self):
        return
    def get_length_distribution(self,df_speech):
        plt.figure(figsize=(10,6))
        doc_lens = [len(d) for d in df_speech.content]
        plt.hist(doc_lens, bins = 100)
        plt.title('Distribution of Question character length')
        plt.ylabel('Number of questions')
        plt.xlabel('Question character length')
        sns.despine();
    
               
        


