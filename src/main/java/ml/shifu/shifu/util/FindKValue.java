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

import java.util.Comparator;
import java.util.List;

/**
 * Find the k largest number in List,
 * <p>
 * Warning: this FindKValue would change the order of list
 */
public class FindKValue {

    public static <T> T find(List<T> values, int k, Comparator<T> comparator) {

        if(values == null || values.size() <= k) {
            return null;
        }

        if(values.size() == 1) {
            return values.get(0);
        }

        return quickfind(values, k, comparator, 0, values.size() - 1);
    }

    private static <T> T quickfind(List<T> values, int k, Comparator<T> comparator, int low, int high) {

        int pivotIndex = partition(values, comparator, low, high);

        if(pivotIndex == k)
            return values.get(pivotIndex);

        if(pivotIndex < k) {
            return quickfind(values, k, comparator, pivotIndex + 1, high);
        } else {
            return quickfind(values, k, comparator, low, pivotIndex - 1);
        }
    }

    private static <T> int partition(List<T> values, Comparator<T> comparator, int low, int high) {
        T pivot = values.get(low);
        int i = low;

        for(int j = (low + 1); j <= high; j++) {
            if(comparator.compare(values.get(j), pivot) <= 0) {
                i++;
                if(i < j) {
                    exchange(values, i, j);
                }
            }
        }
        exchange(values, low, i);

        return i;
    }

    private static <T> void exchange(List<T> values, int i, int j) {
        T tmp = values.get(i);
        values.set(i, values.get(j));
        values.set(j, tmp);
    }
}
