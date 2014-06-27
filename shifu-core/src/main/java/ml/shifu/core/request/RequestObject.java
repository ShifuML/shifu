package ml.shifu.core.request;

import com.fasterxml.jackson.annotation.JsonIgnore;
import ml.shifu.core.util.Params;

import java.util.Map;

public class RequestObject {


    public static enum ExecutionMode {
        LOCAL_SINGLE, LOCAL_CONCURRENT, HADOOP
    }

    private String requestType;
    private ExecutionMode executionMode;
    private Params globalParams = null;


    //private Params fieldParams = null;
    private Map<String, Params> fieldParamsMap;

    public Params getGlobalParams() {

        return globalParams;
    }

    public void setGlobalParams(Params globalParams) {
        this.globalParams = globalParams;
    }

    @JsonIgnore
    public Params getFieldParams(String fieldName) {
        if (fieldParamsMap.containsKey(fieldName)) {
            return fieldParamsMap.get(fieldName);
        } else if (fieldParamsMap.containsKey("$$default")) {
            return fieldParamsMap.get("$$default");
        } else {
            throw new RuntimeException("No such field: [" + fieldName + "] and no default value specified");
        }
    }

    public Map<String, Params> getFieldParamsMap() {
        return fieldParamsMap;
    }

    public void setFieldParamsMap(Map<String, Params> fieldParamsMap) {
        this.fieldParamsMap = fieldParamsMap;
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


}