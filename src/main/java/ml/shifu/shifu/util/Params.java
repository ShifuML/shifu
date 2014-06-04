package ml.shifu.shifu.util;

import java.util.Map;
import java.util.HashMap;

public class Params {



    private Map<String, Object> paramMap = new HashMap<String, Object>();

    public void set(String key, Object value) {
        paramMap.put(key, value);
    }

    public Object get(String key) {
        if (paramMap.containsKey(key)) {
            return paramMap.get(key);
        } else {
            throw new RuntimeException("No such param: " + key);
        }
    }

    public Object get(String key, Object defaultValue) {
        if (paramMap.containsKey(key)) {
            return paramMap.get(key);
        } else {
            return defaultValue;
        }
    }

    public Map<String, Object> getParamMap() {
        return paramMap;
    }

    public void setParamMap(Map<String, Object> paramMap) {
        this.paramMap = paramMap;
    }



}
