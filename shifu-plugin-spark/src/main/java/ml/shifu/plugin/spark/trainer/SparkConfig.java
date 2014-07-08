package ml.shifu.plugin.spark.trainer;

import java.util.LinkedHashMap;
import java.util.Map;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class SparkConfig {
    private String master = "local";
    private String appName = "Shifu";
    private Map<String, String> sparkConfig = new LinkedHashMap<String, String>();

    
    
    public Map<String, String> getSparkConfig() {
        if (!sparkConfig.containsKey(master))
            sparkConfig.put("spark.master", master);
        if (!sparkConfig.containsKey(appName))
            sparkConfig.put("spark.app.name", appName);
        return sparkConfig;
    }

    public void setSparkConfig(Map<String, String> sparkConfig) {
        this.sparkConfig = sparkConfig;
        if (!sparkConfig.containsKey(master))
            sparkConfig.put("spark.master", master);
        if (!sparkConfig.containsKey(appName))
            sparkConfig.put("spark.app.name", appName);

    }

}
