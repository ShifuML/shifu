package ml.shifu.core.container;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;

@JsonIgnoreProperties(ignoreUnknown = true)
public class SVMParams {

	private String kernel;
	private double constant;
	private double splitRatio;
	private double gamma;

	public double getGamma() {
		return gamma;
	}

	public void setGamma(double gamma) {
		this.gamma = gamma;
	}

	public String getKernel() {
		return kernel;
	}

	public void setKernel(String kernel) {
		this.kernel = kernel;
	}

	public double getConstant() {
		return constant;
	}

	public void setConstant(double constant) {
		this.constant = constant;
	}

	public double getSplitRatio() {
		return splitRatio;
	}

	public void setSplitRatio(double splitRatio) {
		this.splitRatio = splitRatio;
	}

}
