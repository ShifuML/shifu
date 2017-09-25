/*
 * Copyright [2013-2015] PayPal Software Foundation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package ml.shifu.shifu.core.dtrain.dt;

/**
 * Loss computation for gradient boost decision tree.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
public interface Loss {

    public float computeGradient(float predict, float label);

    public float computeError(float predict, float label);

}

/**
 * Using Math.abs to compute error.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
class AbsoluteLoss implements Loss {

    @Override
    public float computeGradient(float predict, float label) {
        return Float.compare(label, predict) < 0 ? 1f : -1f;
    }

    @Override
    public float computeError(float predict, float label) {
        return Math.abs(label - predict);
    }

}

/**
 * Squared error is used to compute error.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
class SquaredLoss implements Loss {

    @Override
    public float computeGradient(float predict, float label) {
        return 2f * (predict - label);
    }

    @Override
    public float computeError(float predict, float label) {
        float error = predict - label;
        return error * error;
    }

}

/**
 * Squared error is used to compute error. For gradient, half of gradient is using instead of 2 * (y-p).
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
class HalfGradSquaredLoss extends SquaredLoss {

    @Override
    public float computeGradient(float predict, float label) {
        return (predict - label);
    }

}

/**
 * Log function is used for {@link LogLoss}.
 * 
 * @author Zhang David (pengzhang@paypal.com)
 */
class LogLoss implements Loss {
    // reference https://statweb.stanford.edu/~jhf/ftp/trebst.pdf
    @Override
    public float computeGradient(float predict, float label) {
        return (2 - 4 * label) / (float) Math.exp(4 * label * predict - 2 * predict);
    }

    @Override
    public float computeError(float predict, float label) {
        return (float) Math.log1p(1 + Math.exp(2 * predict - 4 * predict * label));
    }
}
