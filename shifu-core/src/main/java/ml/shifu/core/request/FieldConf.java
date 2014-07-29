package ml.shifu.core.request;


import ml.shifu.core.util.Params;

public class FieldConf {

    private String namePattern;
    private Params params;
    private Binding binding;

    public Binding getBinding() {
        return binding;
    }

    public void setBinding(Binding binding) {
        this.binding = binding;
    }


    public String getNamePattern() {
        return namePattern;
    }

    public void setNamePattern(String namePattern) {
        this.namePattern = namePattern;
    }

    public Params getParams() {
        return params;
    }

    public void setParams(Params params) {
        this.params = params;
    }


}
