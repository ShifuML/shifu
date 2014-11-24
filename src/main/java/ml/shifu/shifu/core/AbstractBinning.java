package ml.shifu.shifu.core;

import java.util.List;

/**
 * Created by yliu15 on 2014/11/20.
 */
public interface AbstractBinning<T> {

    public void clearBins();
    public void add(T e);
    public List<T> getBins();
}
