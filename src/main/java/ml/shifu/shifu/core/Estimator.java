package ml.shifu.shifu.core;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class Estimator<T extends Comparable<T>> {

    private static final long MAX_TOT_ELEMS = 1024L * 1024L * 1024L * 1024L;

    private final List<List<T>> buffer = new ArrayList<List<T>>();
    private final int numBin;
    private final int maxElementsPerBuffer;
    private int totalElements;
    private T min;
    private T max;

    public Estimator(int numBin) {
        this.numBin = numBin;
        this.maxElementsPerBuffer = computeMaxElementsPerBuffer();
    }

    private int computeMaxElementsPerBuffer()
    {
        double epsilon = 1.0 / (numBin - 1.0);
        int b = 2;
        while ((b - 2) * (0x1L << (b - 2)) + 0.5 <= epsilon * MAX_TOT_ELEMS) {
            ++b;
        }
        return (int) (MAX_TOT_ELEMS / (0x1L << (b - 1)));
    }


    private void collapse(List<T> a, List<T> b, List<T> out)
    {
        int indexA = 0, indexB = 0, count = 0;
        T smaller = null;
        while (indexA < maxElementsPerBuffer || indexB < maxElementsPerBuffer) {
            if (indexA >= maxElementsPerBuffer ||
                    (indexB < maxElementsPerBuffer && a.get(indexA).compareTo(b.get(indexB)) >= 0 )) {
                smaller = b.get(indexB++);
            } else {
                smaller = a.get(indexA++);
            }

            if (count++ % 2 == 0) {
                out.add(smaller);
            }
        }
        a.clear();
        b.clear();
    }

    private void ensureBuffer(int level)
    {
        while (buffer.size() < level + 1) {
            buffer.add(null);
        }
        if (buffer.get(level) == null) {
            buffer.set(level, new ArrayList<T>());
        }
    }

    private void recursiveCollapse(List<T> buf, int level)
    {
        ensureBuffer(level + 1);

        List<T> merged;
        if (buffer.get(level + 1).isEmpty()) {
            merged = buffer.get(level + 1);
        } else {
            merged = new ArrayList<T>(maxElementsPerBuffer);
        }

        collapse(buffer.get(level), buf, merged);
        if (buffer.get(level + 1) != merged) {
            recursiveCollapse(merged, level + 1);
        }
    }

    public void add(T elem)
    {
        if (totalElements == 0 || elem.compareTo(min) < 0) {
            min = elem;
        }
        if (totalElements == 0 || max.compareTo(elem) < 0) {
            max = elem;
        }

        if (totalElements > 0 && totalElements % (2 * maxElementsPerBuffer) == 0) {
            Collections.sort(buffer.get(0));
            Collections.sort(buffer.get(1));
            recursiveCollapse(buffer.get(0), 1);
        }

        ensureBuffer(0);
        ensureBuffer(1);
        int index = buffer.get(0).size() < maxElementsPerBuffer ? 0 : 1;
        buffer.get(index).add(elem);
        totalElements++;
    }

    public void clear()
    {
        buffer.clear();
        totalElements = 0;
    }

    public List<T> getBin()
    {
        List<T> quantiles = new ArrayList<T>();
        quantiles.add(min);

        if (buffer.size() == 0) {
            quantiles.add(max);
            return quantiles;
        }

        if (buffer.get(0) != null) {
            Collections.sort(buffer.get(0));
        }
        if (buffer.get(1) != null) {
            Collections.sort(buffer.get(1));
        }

        int[] index = new int[buffer.size()];
        long S = 0;
        for (int i = 1; i <= this.numBin - 2; i++) {
            long targetS = (long) Math.ceil(i * (totalElements / (numBin - 1.0)));

            while (true) {
                T smallest = max;
                int minBufferId = -1;
                for (int j = 0; j < buffer.size(); j++) {
                    if (buffer.get(j) != null && index[j] < buffer.get(j).size()) {
                        if (smallest.compareTo(buffer.get(j).get(index[j])) >= 0) {
                            smallest = buffer.get(j).get(index[j]);
                            minBufferId = j;
                        }
                    }
                }

                long incrementS = minBufferId <= 1 ? 1L : (0x1L << (minBufferId - 1));
                if (S + incrementS >= targetS) {
                    quantiles.add(smallest);
                    break;
                } else {
                    index[minBufferId]++;
                    S += incrementS;
                }
            }
        }

        quantiles.add(max);
        return quantiles;
    }
}

