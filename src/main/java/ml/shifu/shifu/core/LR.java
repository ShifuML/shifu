package ml.shifu.shifu.core;

import java.util.Arrays;

import org.apache.commons.lang.StringUtils;
import org.encog.mathutil.BoundMath;
import org.encog.ml.BasicML;
import org.encog.ml.data.MLData;
import org.encog.ml.data.basic.BasicMLData;

public class LR extends BasicML{

    /**
     * Serial id.
     */
    private static final long serialVersionUID = 1L;
    
    private double[] weights;
    
    public LR(double[] weights){
        this.weights = weights;
    }
    
    public final MLData compute(final MLData input) {
        MLData result = new BasicMLData(1);
        double score = this.sigmoid(input.getData(),this.weights);
        result.setData(0, score);
        return result;
    }
    
    public int getInputCount(){
        return this.weights.length;
    }
    
    @Override
    public String toString(){
        return Arrays.toString(this.weights);
    }
    
    /**
     * Compute sigmoid value by dot operation of two vectors.
     */
    private double sigmoid(double[] inputs, double[] weights) {
        double value = 0.0d;
        for(int i = 0; i < weights.length; i++) {
            value += weights[i] * inputs[i];
        }

        return 1.0d / (1.0d + BoundMath.exp(-1 * value));
    }
    
    
    @Override
    public void updateProperties(){
        
    }
    
    public static LR loadFromString(String input){
        String target = StringUtils.remove(StringUtils.remove(input, '['),']');
        String[] ws = target.split(",");
        double[] weights = new double[ws.length];
        int index = 0;
        for(String weight:ws){
            weights[index++] = Double.parseDouble(weight);
        }
        return new LR(weights);
    }
    
    public static void main(String[] args){
        String input = "[-0.10278490719596094, -0.23274653424075714, 0.11381861377262863, 0.1810198234175508, 0.4072712673250277, 0.5831561865840784, 0.3941112705052633, 0.09391158682896095, -0.28126099080721884, 0.32767652326554414, -0.21705444827736756, 0.19049676260826665, 0.01689999845178048, 0.14799298375376144, 0.20825502093731227, -0.11147709102620681, 0.1562122312346503, -0.08979543838063228, 0.03807067691509311, 0.22425803906215244, 0.17362197694508907, 0.2797724493155119, 0.7848803213613736, -0.047986511023306234, -0.23062593860512265, 0.1289474938348541, -0.6680192405847827, 0.11585690554670504, 0.28601457410729597]";
        LR lr = LR.loadFromString(input);
        System.out.print(lr);
        
    }

}
