package ml.shifu.plugin.spark.stats.interfaces;

import java.util.List;

import org.dmg.pmml.ModelStats;

import com.google.common.base.Splitter;

/*
 * This class holds an array of states for every column in the table.
 * Two instances exist for Univariate and Binomial Stats.
 */

public abstract class ColumnStateArray implements java.io.Serializable {

    private static final long serialVersionUID = 1L;
    protected List<ColumnState> states;
    protected String delimiter;
        
    abstract public ColumnStateArray getNewBlank() throws Exception;
    
    public ColumnStateArray merge(ColumnStateArray stateArray2) throws Exception {
        checkClass(stateArray2);
        if(stateArray2.getStateArray().size() != states.size())
            throw new Exception("Sizes of state arrays don't match");
        int index= 0;
        for(ColumnState colState: stateArray2.getStateArray()) {
            states.get(index).merge(colState);
            index++;
        }
        return this;
    }
    
    public abstract void checkClass(ColumnStateArray stateArray) throws Exception;

    public void addData(String line) {
        int index= 0;
        for(String strValue: Splitter.on(delimiter).split(line)) {
            states.get(index).addData(strValue);
            index++;
        }
        return;
    }
    
    public List<ColumnState> getStateArray() {
        return states;
    }

    public ModelStats getModelStats() {
        ModelStats modelStats= new ModelStats();
        for(ColumnState state: this.states) {
            modelStats.withUnivariateStats(state.getUnivariateStats());
        }
        
        return modelStats;
    }

}
