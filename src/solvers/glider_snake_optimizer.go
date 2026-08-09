package solvers

import (
	"algorithmia/src/functions"
	"algorithmia/src/models"
	"fmt"
	"math"
	"math/rand/v2"
)

const (
	gliderSnakeOptimizerPopulation   = 20
	gliderSnakeOptimizerMaxSteps     = 1000
	gliderSnakeOptimizerTolerance    = 1e-6
	gliderSnakeOptimizerRange        = 5.0
	gliderSnakeOptimizerLevyLambda   = 1.5
	gliderSnakeOptimizerPerturbScale = 0.1
	gliderSnakeOptimizerMatingThresh = 0.6
)

// GliderSnakeOptimizer minimizuje konfigurisanu benchmark funkciju koristeći algoritam optimizacije zmijom:
// populacija se deli na ženske i muške: u fazi parenja ženske lete Levy-flight-om ka najboljem a muški ka najboljoj
// ženskoj, u fazi hrane sve zmije konvergiraju ka najboljem uz eksponencijalno prigušenje i pohlepno prihvatanje
// problem.Point inicijalizuje populaciju i određuje njenu dimenzionalnost
func GliderSnakeOptimizer(problem models.Problem) (result models.Result, err error) {

	if len(problem.Point) == 0 {
		err = fmt.Errorf("starting point is required")
		return result, err
	}

	fnName, _ := problem.Payload["function"].(string)
	fn, err := functions.Get(fnName)
	if err != nil {
		return result, err
	}
	if err := fn.ValidateDimension(len(problem.Point)); err != nil {
		return result, err
	}

	population := gliderSnakeOptimizerPopulation
	if v, ok := problem.Payload["population"].(float64); ok {
		population = int(v)
	}

	maxSteps := gliderSnakeOptimizerMaxSteps
	if v, ok := problem.Payload["max_steps"].(float64); ok {
		maxSteps = int(v)
	}

	tolerance := gliderSnakeOptimizerTolerance
	if v, ok := problem.Payload["tolerance"].(float64); ok {
		tolerance = v
	}
	includeTrajectory, _ := problem.Payload["include_trajectory"].(bool)
	var trajectory []models.TrajectoryPoint

	dimensions := len(problem.Point)
	half := population / 2

	positions := make([][]float64, population)
	values := make([]float64, population)
	for i := range positions {
		position := make([]float64, dimensions)
		for d := range position {
			position[d] = (rand.Float64()*2 - 1) * gliderSnakeOptimizerRange
		}
		positions[i] = position
		values[i] = fn.Evaluate(position)
	}

	best := append([]float64(nil), positions[0]...)
	bestValue := values[0]
	for i, val := range values {
		if val < bestValue {
			bestValue = val
			best = append([]float64(nil), positions[i]...)
		}
	}

	maxNoImprove := max(50, maxSteps/20)

	temperature := 1.0
	noImprove := 0
	steps := 0
	for ; steps < maxSteps; steps++ {

		prevBestValue := bestValue

		if temperature > gliderSnakeOptimizerMatingThresh {
			bestFemale := 0
			for i := 1; i < half; i++ {
				if values[i] < values[bestFemale] {
					bestFemale = i
				}
			}

			for i := 0; i < half; i++ {
				newPos := make([]float64, dimensions)
				levyStep := mantegnaLevy(gliderSnakeOptimizerLevyLambda)
				for d := range newPos {
					newPos[d] = positions[i][d] + levyStep*(best[d]-positions[i][d]) + (rand.Float64()*2-1)*gliderSnakeOptimizerPerturbScale
				}
				newValue := fn.Evaluate(newPos)
				if newValue < values[i] {
					positions[i] = newPos
					values[i] = newValue
					if newValue < bestValue {
						bestValue = newValue
						best = append([]float64(nil), newPos...)
					}
				}
			}

			for i := half; i < population; i++ {
				newPos := make([]float64, dimensions)
				for d := range newPos {
					newPos[d] = positions[i][d] + rand.Float64()*(positions[bestFemale][d]-positions[i][d])
				}
				newValue := fn.Evaluate(newPos)
				if newValue < values[i] {
					positions[i] = newPos
					values[i] = newValue
					if newValue < bestValue {
						bestValue = newValue
						best = append([]float64(nil), newPos...)
					}
				}
			}
		} else {
			for i := 0; i < population; i++ {
				newPos := make([]float64, dimensions)
				factor := (rand.Float64()*2 - 1) * math.Exp(-math.Abs(values[i]-bestValue)/(values[i]+1e-10))
				for d := range newPos {
					newPos[d] = best[d] + factor*(positions[i][d]-best[d])
				}
				newValue := fn.Evaluate(newPos)
				if newValue < values[i] {
					positions[i] = newPos
					values[i] = newValue
					if newValue < bestValue {
						bestValue = newValue
						best = append([]float64(nil), newPos...)
					}
					if includeTrajectory {
						trajectory = recordTrajectory(trajectory, steps, best, bestValue, false)
					}
				}
			}
		}

		temperature -= 1.0 / float64(maxSteps)

		if math.Abs(bestValue-prevBestValue) < tolerance {
			noImprove++
		} else {
			noImprove = 0
		}

		if noImprove >= maxNoImprove {
			steps++
			break
		}
	}

	if includeTrajectory {
		trajectory = recordTrajectory(trajectory, steps, best, bestValue, true)
		trajectory = capTrajectory(trajectory, trajectoryCapLen)
	}

	result = models.Result{
		Method:     "glider_snake_optimizer",
		Point:      best,
		Value:      bestValue,
		Steps:      steps,
		Function:   fn.Name,
		Trajectory: trajectory,
	}

	return result, nil
}

// mantegnaLevy uzorkuje korak iz Levy raspodele koristeći Mantegna algoritam
// sa parametrom lambda, za simuliranje dugih skokova tokom pretrage
func mantegnaLevy(lambda float64) float64 {

	numerator := math.Gamma(1+lambda) * math.Sin(math.Pi*lambda/2)
	denominator := math.Gamma((1+lambda)/2) * lambda * math.Pow(2, (lambda-1)/2)
	sigma := math.Pow(numerator/denominator, 1/lambda)

	u := randNorm() * sigma
	v := randNorm()

	return u / math.Pow(math.Abs(v), 1/lambda)
}
