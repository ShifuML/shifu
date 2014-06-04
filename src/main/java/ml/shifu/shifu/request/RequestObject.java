package ml.shifu.shifu.request;

import ml.shifu.shifu.util.Params;

import java.util.Map;

public class RequestObject {


    public static enum ExecutionMode {
        LOCAL_SINGLE, LOCAL_CONCURRENT, HADOOP
    }

    private String action;
    private ExecutionMode executionMode;
    private Map<String, String> bindings;
    private Map<String, Object> parameters;
    private Params params = null;

    public Params getParams() {
        if (params == null) {
            params = new Params();
            params.setParamMap(parameters);
        }
        return params;
    }

    public String getAction() {
        return action;
    }

    public void setAction(String action) {
        this.action = action;
    }

    public ExecutionMode getExecutionMode() {
        return executionMode;
    }

    public void setExecutionMode(ExecutionMode executionMode) {
        this.executionMode = executionMode;
    }

    public Map<String, String> getBindings() {
        return bindings;
    }

    public void setBindings(Map<String, String> bindings) {
        this.bindings = bindings;
    }

    public Map<String, Object> getParameters() {
        return parameters;
    }

   // public Params getParams() {

   // }

    public void setParameters(Map<String, Object> parameters) {
        this.parameters = parameters;
    }

}
