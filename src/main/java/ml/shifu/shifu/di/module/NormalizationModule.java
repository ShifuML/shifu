package ml.shifu.shifu.di.module;


import com.google.inject.AbstractModule;
import ml.shifu.shifu.di.spi.Normalizer;

public class NormalizationModule extends AbstractModule {

    private Class normalizerImplClass;

    public NormalizationModule(String className) {
        try {
            normalizerImplClass = Class.forName(className);
        } catch (Exception e) {
            throw new RuntimeException("No such implementation class: " + className);
        }
    }

    @Override
    protected void configure() {
        bind(Normalizer.class).to(normalizerImplClass);
    }
}
