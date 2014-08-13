package ml.shifu.core.util;

import com.fasterxml.jackson.annotation.JsonIgnore;
import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.core.request.FieldConf;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

public class Params extends LinkedHashMap<String, Object> {


    private Map<String, Params> fieldParamsMap = null;
    private Map<String, String> bindings = null;
    private List<FieldConf> fieldConfs = null;

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
    public FieldConf getFieldConfig(String fieldNameString) throws Exception {

        if (this.fieldConfs == null) {
            fieldConfs = new ArrayList<FieldConf>();
            ObjectMapper jsonMapper = new ObjectMapper();

            for (Object field : (List<Object>) this.get("fields")) {
                FieldConf fieldConf = jsonMapper.readValue(jsonMapper.writeValueAsString(field), FieldConf.class);
                fieldConfs.add(fieldConf);
            }

        }

        FieldConf defaultFieldConf = null;

        for (FieldConf fieldConf : fieldConfs) {
            if (fieldConf.getNamePattern().equals("$default")) {
                defaultFieldConf = fieldConf;
            }

            if (fieldConf.getNamePattern().equals(fieldNameString)) {
                return fieldConf;
            }


        }

        if (defaultFieldConf != null) {
            return defaultFieldConf;
        } else {
            throw new RuntimeException("No such field: " + fieldNameString + ", and no default params provided");
        }
    }

    @JsonIgnore
    public Params getFieldParams(String fieldNameString) throws Exception {

        if (this.fieldParamsMap == null) {
            fieldParamsMap = new LinkedHashMap<String, Params>();
            ObjectMapper jsonMapper = new ObjectMapper();

            for (Object field : (List<Object>) this.get("fields")) {
                FieldConf fieldConf = jsonMapper.readValue(jsonMapper.writeValueAsString(field), FieldConf.class);
                fieldParamsMap.put(fieldConf.getNamePattern(), fieldConf.getParams());
            }

        }


        //TODO: add pattern matching
        if (fieldParamsMap.containsKey(fieldNameString)) {
            return fieldParamsMap.get(fieldNameString);
        } else if (fieldParamsMap.containsKey("$default")) {
            return fieldParamsMap.get("$default");
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



}
