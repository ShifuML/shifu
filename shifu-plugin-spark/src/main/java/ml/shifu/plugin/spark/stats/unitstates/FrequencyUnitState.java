package ml.shifu.plugin.spark.stats.unitstates;

import org.dmg.pmml.Counts;
import org.dmg.pmml.UnivariateStats;

import ml.shifu.core.util.Params;
import ml.shifu.plugin.spark.stats.interfaces.UnitState;

public class FrequencyUnitState implements UnitState {

    private static final long serialVersionUID = 1L;
    private double totalFreq = 0;
    private double missingFreq = 0;
    private double invalidFreq = 0;

    public FrequencyUnitState() {
        this.totalFreq= 0.0;
        this.missingFreq= 0.0;
        this.invalidFreq= 0.0;
    }
    
    public UnitState getNewBlank() {
        return new FrequencyUnitState();
    }

    public void merge(UnitState state) throws Exception {
        if(!(state instanceof FrequencyUnitState))
            throw new Exception("Expected FrequencyUnitState, got " + state.getClass().toString());
        FrequencyUnitState newState= (FrequencyUnitState) state;
        this.totalFreq+= newState.getTotal();
        this.missingFreq+= newState.getMissing();
        this.invalidFreq+= newState.getInvalid();
    }

    public void addData(Object objValue) {
        this.totalFreq++;
        // check if double, for optimization 
        if(objValue instanceof Double)
            return;
        
        if(objValue==null || objValue.toString().length()==0)
            this.missingFreq++;
        else if(!isNumeric(objValue.toString()))   
            this.invalidFreq++;            
    }

    public double getInvalid() {
        return this.invalidFreq;
    }
    public double getMissing() {
        return this.missingFreq;
    }
    public double getTotal() {
        return this.totalFreq;
    }
    
    
    private boolean isNumeric(String str)  
    {  
        try  {  
            Double.parseDouble(str);  
        }  catch(NumberFormatException e)  {
            return false;  
        }  
        return true;  
    }
    
    public void populateUnivariateStats(UnivariateStats univariateStats, Params params) {
        Counts counts= univariateStats.getCounts();
        if(counts==null) {
            counts= new Counts();
        }
        counts.withInvalidFreq(this.invalidFreq);
        counts.withMissingFreq(this.missingFreq);
        counts.withTotalFreq(this.totalFreq);        
        
        univariateStats.withCounts(counts);
    }
    
}
