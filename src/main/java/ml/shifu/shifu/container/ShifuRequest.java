package ml.shifu.shifu.container;


import ml.shifu.shifu.request.RequestObject;

import java.util.List;

public class ShifuRequest {

    String version;
    String description;

    public List<RequestObject> getRequests() {
        return requests;
    }

    public void setRequests(List<RequestObject> requests) {
        this.requests = requests;
    }

    public String getVersion() {
        return version;
    }

    public void setVersion(String version) {
        this.version = version;
    }

    public String getDescription() {
        return description;
    }

    public void setDescription(String description) {
        this.description = description;
    }

    List<RequestObject> requests;

}
