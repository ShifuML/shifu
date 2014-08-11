package ml.shifu.core.container;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.HashMap;
import java.util.Map;

@JsonIgnoreProperties(ignoreUnknown = true)
public class ColumnDerivedResult {
    private Map<String, Object> userDefined = new HashMap<String, Object>();

    public Map<String, Object> getUserDefined() {
        return userDefined;
    }

    public void setUserDefined(Map<String, Object> userDefined) {
        this.userDefined = userDefined;
    }


}
