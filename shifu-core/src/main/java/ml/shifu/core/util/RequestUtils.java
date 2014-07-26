package ml.shifu.core.util;

import com.fasterxml.jackson.databind.ObjectMapper;
import ml.shifu.core.request.Binding;
import ml.shifu.core.request.FieldConf;
import ml.shifu.core.request.Request;

import java.util.ArrayList;
import java.util.List;

public class RequestUtils {

    public static Binding getUniqueBinding(Request req, String spi) {
        return getUniqueBinding(req, spi, false);
    }


    public static Binding getUniqueBinding(Request req, String spi, Boolean required) {
        Binding binding = null;

        for (Binding b : req.getBindings()) {
            if (b.getSpi().equals(spi)) {
                if (binding == null) {
                    binding = b;
                } else {
                    throw new RuntimeException("Binding should be unique: " + spi);
                }
            }
        }


        if (required && binding == null) {
            throw new RuntimeException("Missing binding for: " + spi);
        }

        return binding;
    }

    public static Binding getBindingBySpi(Request req, String spi) {
        for (Binding b : req.getBindings()) {
            if (b.getSpi().equals(spi)) {
                return b;
            }

        }
        return null;
        //throw new RuntimeException("Missing binding for: " + spi);
    }

    public static Params getBindingParamsBySpi(Request req, String spi) {
        return getBindingBySpi(req, spi).getParams();
    }

    public static FieldConf getFieldConfByName(List<FieldConf> fields, String name) {

        FieldConf defaultFieldConf = null;

        for (FieldConf fieldConf : fields) {
            if (matchNamePattern(name, fieldConf.getNamePattern())) {
                return fieldConf;
            }

            if (fieldConf.getNamePattern().equals("$default")) {
                defaultFieldConf = fieldConf;
            }
        }

        if (defaultFieldConf != null) {
            return defaultFieldConf;
        }

        return null;
    }

    public static Boolean matchNamePattern(String name, String namePattern) {
        //TODO: add pattern matching
        return name.equals(namePattern);
    }

    public static List<FieldConf> getFieldConfs(Params params) throws Exception {
        List<FieldConf> fieldConfs = new ArrayList<FieldConf>();
        ObjectMapper jsonMapper = new ObjectMapper();

        for (Object field : (List<Object>) params.get("fields")) {
            FieldConf fieldConf = jsonMapper.readValue(jsonMapper.writeValueAsString(field), FieldConf.class);
            fieldConfs.add(fieldConf);
        }

        return fieldConfs;
    }

}
