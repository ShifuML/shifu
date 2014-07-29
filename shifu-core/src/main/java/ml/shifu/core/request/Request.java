package ml.shifu.core.request;


import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public class Request {

    private String name;
    private String description;
    private Binding processor;
    private List<Binding> bindings;

    public String getName() {
        return name;
    }

    public void setName(String name) {
        this.name = name;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    public Binding getProcessor() {
        return processor;
    }

    public void setProcessor(Binding processor) {
        this.processor = processor;
    }

    public List<Binding> getBindings() {
        return bindings;
    }

    public void setBindings(List<Binding> bindings) {
        this.bindings = bindings;
    }


}