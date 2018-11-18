#!/usr/bin/env python3
# encoding: utf-8
"""
@author: Daniel Fuchs

CS3001: Data Science - Homework #7: K-Means
"""
import numpy as np
from kmeans import kmeans, printTable, showClusters2D, loadCSV


def main():
    # PART 1
    football_data = loadCSV('football.csv')
    part1_q1(football_data, show_graph=False)
    part1_q2(football_data, show_graph=False)
    part1_q3(football_data, show_graph=False)
    part1_q4(football_data, show_graph=False)

    # PART 2
    iris_data = loadCSV('iris.csv')
    standard_point = [('setosa', 4.4, 2.9, 1.4, 0.2),
                      ('versicolor', 4.9, 2.4, 3.3, 1.0),
                      ('virginica', 5.8, 2.8, 5.1, 2.4)]
    euclid_cluster = kmeans(iris_data, 3, distance_type='Euclidean', initCentroids=standard_point)
    cosine_cluster = kmeans(iris_data, 3, distance_type='Cosine', initCentroids=standard_point)
    jaccard_cluster = kmeans(iris_data, 3, distance_type='Jaccard', initCentroids=standard_point)
    sse_list = [compute_SSE(euclid_cluster),
                compute_SSE(cosine_cluster),
                compute_SSE(jaccard_cluster)]
    acc_list = [compute_accuracy(euclid_cluster),
                compute_accuracy(cosine_cluster),
                compute_accuracy(jaccard_cluster)]
    itr_list = [euclid_cluster["iterations"],
                cosine_cluster["iterations"],
                jaccard_cluster["iterations"]]
    part2_q1(sse_list)
    part2_q2(acc_list)
    part2_q3(itr_list)
    return 0


def part1_q1(dataset, show_graph=False):
    print(">>> PART 1: QUESTION 1")
    clustering = kmeans(dataset, 2, initCentroids=[('i1', 4, 6), ('i2', 5, 4)], distance_type='Manhattan')
    printTable(clustering["centroids"])
    if show_graph:
        showClusters2D(clustering)
    print('')


def part1_q2(dataset, show_graph=False):
    print(">>> PART 1: QUESTION 2")
    clustering = kmeans(dataset, 2, initCentroids=[('i1', 4, 6), ('i2', 5, 4)], distance_type='Euclidean')
    printTable(clustering["centroids"])
    if show_graph:
        showClusters2D(clustering)
    print('')


def part1_q3(dataset, show_graph=False):
    print(">>> PART 1: QUESTION 3")
    clustering = kmeans(dataset, 2, initCentroids=[('i1', 3, 3), ('i2', 8, 3)], distance_type='Manhattan')
    printTable(clustering["centroids"])
    if show_graph:
        showClusters2D(clustering)
    print('')


def part1_q4(dataset, show_graph=False):
    print(">>> PART 1: QUESTION 4")
    clustering = kmeans(dataset, 2, initCentroids=[('i1', 3, 2), ('i2', 4, 8)], distance_type='Manhattan')
    printTable(clustering["centroids"])
    if show_graph:
        showClusters2D(clustering)
    print('')


def part2_q1(stats):
    print(">>> PART 2: QUESTION 1")
    print("Euclidean SSE Value:", stats[0])
    print("   Cosine SSE Value:", stats[1])
    print("  Jaccard SSE Value:", stats[2])
    print('')


def part2_q2(stats):
    print(">>> PART 2: QUESTION 2")
    print("Euclidean Accuracy:", stats[0])
    print("   Cosine Accuracy:", stats[1])
    print("  Jaccard Accuracy:", stats[2])
    print('')


def part2_q3(stats):
    print(">>> PART 2: QUESTION 3")
    print("Euclidean Iterations:", stats[0])
    print("   Cosine Iterations:", stats[1])
    print("  Jaccard Iterations:", stats[2])
    print('')


def compute_SSE(cluster_data):
    centroids = cluster_data['centroids']
    clusters = cluster_data['clusters']
    total_error = 0
    for i in range(len(clusters)):
        center_point = centroids[i][1:]
        for point in clusters[i]:
            for j in range(len(point) - 1):
                total_error += ((point[j+1] - center_point[j]) ** 2)
    return total_error


def compute_accuracy(cluster_data):
    labels = ['setosa', 'versicolor', 'virginica']
    clusters = cluster_data['clusters']
    cluster_groups = []
    for i in range(len(clusters)):
        counter = {}
        cluster_groups.append('')
        for j in range(len(labels)):
            counter[labels[j]] = 0
        for point in clusters[i]:
            counter[point[0]] += 1
        cluster_groups[i] = list(sorted(counter.items(), key=lambda x: x[1], reverse=True))[0]

    match = 0
    mismatch = 0
    for i in range(len(clusters)):
        for point in clusters[i]:
            if point[0] == cluster_groups[i][0]:
                match += 1
            else:
                mismatch += 1
    return match / (match + mismatch)


if __name__ == '__main__':
    main()
