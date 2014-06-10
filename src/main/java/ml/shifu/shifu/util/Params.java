package ml.shifu.shifu.util;

import java.util.Map;
import java.util.HashMap;

public class Params extends HashMap<String, Object> {



    //private Map<String, Object> paramMap = new HashMap<String, Object>();

    //public void set(String key, Object value) {
    //    super.put(key, value);
    //}

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
   /*
    public Map<String, Object> getParamMap() {
        return paramMap;
    }

    public void setParamMap(Map<String, Object> paramMap) {
        this.paramMap = paramMap;
    }
       */


}
