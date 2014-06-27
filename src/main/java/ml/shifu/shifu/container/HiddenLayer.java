package ml.shifu.shifu.container;

public class HiddenLayer {

    private String activationFunction;
    private Integer numHiddenNodes;


    public String getActivationFunction() {
        return activationFunction;
    }

    public void setActivationFunction(String activationFunction) {
        this.activationFunction = activationFunction;
    }



    public Integer getNumHiddenNodes() {
        return numHiddenNodes;
    }

    public void setNumHiddenNodes(Integer numHiddenNodes) {
        this.numHiddenNodes = numHiddenNodes;
    }

}
