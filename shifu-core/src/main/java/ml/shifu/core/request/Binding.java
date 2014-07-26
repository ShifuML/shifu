package ml.shifu.core.request;

import ml.shifu.core.util.Params;

public class Binding {

    private String spi;
    private String impl;
    private Params params;

    public String getSpi() {
        return spi;
    }

    public void setSpi(String spi) {
        this.spi = spi;
    }

    public String getImpl() {
        return impl;
    }

    public void setImpl(String impl) {
        this.impl = impl;
    }

    public Params getParams() {
        return params;
    }

    public void setParams(Params params) {
        this.params = params;
    }


}
