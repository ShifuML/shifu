package ml.shifu.shifu.request;

import com.fasterxml.jackson.annotation.JsonIgnore;
import ml.shifu.shifu.util.Params;
import java.util.Map;

public class RequestObject {


    public static enum ExecutionMode {
        LOCAL_SINGLE, LOCAL_CONCURRENT, HADOOP
    }



    private String requestName;
    private String requestType;
    private ExecutionMode executionMode;
    private Params params = null;


    public String getRequestName() {
        return requestName;
    }

    public void setRequestName(String requestName) {
        this.requestName = requestName;
    }

    public Params getParams() {

        return params;
    }

    public void setParams(Params params) {
        this.params = params;
    }

    public String getRequestType() {
        return requestType;
    }

    public void setRequestType(String requestType) {
        this.requestType = requestType;
    }

    public ExecutionMode getExecutionMode() {
        return executionMode;
    }

    public void setExecutionMode(ExecutionMode executionMode) {
        this.executionMode = executionMode;
    }

    @JsonIgnore
    public Params getFieldParams(String fieldNameString) {
        return ((Map<String, Params>)params.get("fieldParamsMap")).get(fieldNameString);
    }

}
