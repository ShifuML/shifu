/*
 * Copyright [2012-2014] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.util;

import org.apache.commons.collections.CollectionUtils;

import java.util.Comparator;
import java.util.List;

/**
 * QuickSort class
 */
public class QuickSort {

    /*
     * sort data by @Comparator
     * 
     * @param values - list of data
     * 
     * @param comparator the comparator
     */
    public static <T> void sort(List<T> values, Comparator<T> comparator) {
        if(CollectionUtils.isEmpty(values)) {
            return;
        }

        quicksort(values, comparator, 0, values.size() - 1);
    }

    /*
     * sort the data that implements Comparable
     * 
     * @param values
     *            - list of data
     */
    public static <T extends Comparable<T>> void sort(List<T> values) {
        if(CollectionUtils.isEmpty(values)) {
            return;
        }

        quicksort(values, 0, values.size() - 1);
    }

    /*
     * quick in-place sort
     * 
     * @param values
     */
    private static <T> void quicksort(List<T> values, Comparator<T> comparator, int low, int high) {
        int i = low, j = high;
        // get the pivot element from the middle of the list
        T pivot = values.get(low + (high - low) / 2);

        // divide into two parts by pivot
        while(i <= j) {
            // from the left to find the first element that greater than pivot
            while(comparator.compare(pivot, values.get(i)) > 0) {
                i++;
            }

            // form the right to find the first element that less than pivot
            while(comparator.compare(pivot, values.get(j)) < 0) {
                j--;
            }

            // Wowo, we found those two elements, exchange them
            if(i <= j) {
                exchange(values, i, j);
                i++;
                j--;
            }
        }

        // have data on the left, sort it
        if(low < j) {
            quicksort(values, comparator, low, j);
        }

        // have data on the right, sort it
        if(i < high) {
            quicksort(values, comparator, i, high);
        }
    }

    /*
     * quick in-place sort
     * 
     * @param values
     */
    private static <T extends Comparable<T>> void quicksort(List<T> values, int low, int high) {
        int i = low, j = high;
        // get the pivot element from the middle of the list
        T pivot = values.get(low + (high - low) / 2);

        // divide into two parts by pivot
        while(i <= j) {
            // from the left to find the first element that greater than pivot
            while(pivot.compareTo(values.get(i)) > 0) {
                i++;
            }

            // form the right to find the first element that less than pivot
            while(pivot.compareTo(values.get(j)) < 0) {
                j--;
            }

            // Wowo, we found those two elements, exchange them
            if(i <= j) {
                exchange(values, i, j);
                i++;
                j--;
            }
        }

        // have data on the left, sort it
        if(low < j) {
            quicksort(values, low, j);
        }

        // have data on the right, sort it
        if(i < high) {
            quicksort(values, i, high);
        }
    }

    /*
     * exchange two elements in list
     */
    private static <T> void exchange(List<T> values, int i, int j) {
        T tmp = values.get(i);
        values.set(i, values.get(j));
        values.set(j, tmp);
    }
}
