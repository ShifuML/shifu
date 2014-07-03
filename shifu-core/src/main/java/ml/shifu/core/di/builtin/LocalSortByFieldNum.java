package ml.shifu.core.di.builtin;

import java.util.Collections;
import java.util.Comparator;
import java.util.List;

public class LocalSortByFieldNum {

    public static void sort(List<List<Object>> rows, final int fieldNum) {
        Collections.sort(rows, new Comparator<List<Object>>() {
            @Override
            public int compare(List<Object> a, List<Object> b) {
                return Double.valueOf(a.get(fieldNum).toString()).compareTo(Double.valueOf(b.get(fieldNum).toString()));
            }
        });
    }
}
