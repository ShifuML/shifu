package ml.shifu.shifu.core;

import java.util.*;
import java.util.List;

/**
 * Created by yliu15 on 2014/11/20.
 */
public class ChararrayVarsBinning implements AbstractBinning<String> {

    private Set<String> set;

    public ChararrayVarsBinning(){
        this.set = new HashSet<String>();
    }

    @Override
    public void clearBins() {
        this.set.clear();
    }

    @Override
    public void add(String e) {
        this.set.add(e);
    }

    @Override
    public List<String> getBins() {
        return Arrays.asList(this.set.toArray(new String[1]));
    }
}
