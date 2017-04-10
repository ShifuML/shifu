package ml.shifu.shifu.core.function;

import java.util.function.*;

/**
 * Created by Mark on 3/22/2017.
 */
public abstract class Filter<T, Boolean> extends Function<T, Boolean> {

    abstract Boolean apply(T o);
}
