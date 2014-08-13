package ml.shifu.core.di.builtin;

import ml.shifu.core.di.spi.PMMLCreator;
import ml.shifu.core.util.Params;
import org.dmg.pmml.Application;
import org.dmg.pmml.Header;
import org.dmg.pmml.PMML;


public class ShifuPMMLCreator implements PMMLCreator {

    public PMML create(Params params) {


        PMML pmml = new PMML();

        Application app = new Application();
        app.setName("Shifu");

        Header header = new Header();
        header.setApplication(app);

        pmml.setHeader(header);

        return pmml;
    }
}
