package ml.shifu.core.di.spi;

import ml.shifu.core.request.RequestObject;
import org.dmg.pmml.Model;

public interface ModelElementCreator {

    public Model create(RequestObject req);

}
