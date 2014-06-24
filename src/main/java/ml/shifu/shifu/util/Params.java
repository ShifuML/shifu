package ml.shifu.shifu.util;

import com.fasterxml.jackson.annotation.JsonIgnore;

import java.util.LinkedHashMap;
import java.util.Map;

public class Params extends LinkedHashMap<String, Object> {


    //public void set(String key, Object value) {
    //    super.put(key, value);
    //}

    private Map<String, Params> fieldParamsMap = null;
    private Map<String, String> bindings = null;

    public Map<String, String> getBindings() {
        return bindings;
    }

    public void setBindings(Map<String, String> bindings) {
        this.bindings = bindings;
    }

    public Map<String, Params> getFieldParamsMap() {
        return fieldParamsMap;
    }

    public void setFieldParamsMap(Map<String, Params> fieldParamsMap) {
        this.fieldParamsMap = fieldParamsMap;
    }

    @JsonIgnore
    public Params getFieldParams(String fieldNameString) {
        //TODO: add pattern matching
        if (fieldParamsMap.containsKey(fieldNameString)) {
            return fieldParamsMap.get(fieldNameString);
        } else if (fieldNameString.contains("$$default")) {
            return fieldParamsMap.get("$$default");
        } else {
            throw new RuntimeException("No such field: " + fieldNameString + ", and no default params provided");
        }
    }


    public Object get(String key) {
        if (containsKey(key)) {
            return super.get(key);
        } else {
            throw new RuntimeException("No such param: " + key);
        }
    }

    public Object get(String key, Object defaultValue) {
        if (containsKey(key)) {
            return super.get(key);
        } else {
            return defaultValue;
        }
    }


    public void set(String key, Object value) {
        super.put(key, value);
    }

   /*
    public Map<String, Object> getParamMap() {
        return paramMap;
    }

    public void setParamMap(Map<String, Object> paramMap) {
        this.paramMap = paramMap;
    }
       */


}
