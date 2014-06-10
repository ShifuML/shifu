package ml.shifu.shifu.di.spi;

import ml.shifu.shifu.request.RequestObject;
import org.dmg.pmml.Model;

public interface ModelElementCreator {

    public Model create(RequestObject req);

}
