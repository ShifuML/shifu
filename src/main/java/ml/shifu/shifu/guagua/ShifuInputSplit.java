package ml.shifu.shifu.guagua;

import org.apache.hadoop.mapreduce.lib.input.FileSplit;

import ml.shifu.guagua.hadoop.io.GuaguaInputSplit;

public class ShifuInputSplit extends GuaguaInputSplit {
    private boolean isCrossValidation;
    
    
    public ShifuInputSplit(boolean isMaster,boolean isCrossValidation, FileSplit fileSplit) {
        super(isMaster, fileSplit);
        this.isCrossValidation = isCrossValidation;
    }
    
    public ShifuInputSplit(boolean isMaster, boolean isCrossValidation,FileSplit... fileSplits) {
        super(isMaster,fileSplits);
        this.isCrossValidation = isCrossValidation;
    }

    public boolean isCrossValidation() {
        return isCrossValidation;
    }
    public void setCrossValidation(boolean isCrossValidation) {
        this.isCrossValidation = isCrossValidation;
    }

}
