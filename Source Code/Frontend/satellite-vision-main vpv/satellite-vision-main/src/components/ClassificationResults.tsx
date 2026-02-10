import { motion } from 'framer-motion';
import { TrendingUp, Clock, ImageIcon } from 'lucide-react';
import type { ClassificationResult } from '@/lib/satelliteValidator';

interface ClassificationResultsProps {
  data: Array<{ result: ClassificationResult; preview: string }>;
}

export function ClassificationResults({ data }: ClassificationResultsProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      className="w-full space-y-6"
    >
      <h2 className="text-2xl font-bold text-center mb-6">
        Classification Results ({data.length} image{data.length !== 1 ? 's' : ''})
      </h2>

      {data.map(({ result, preview }, resultIndex) => {
        const topPrediction = result.predictions[0];

        return (
          <motion.div
            key={resultIndex}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: resultIndex * 0.2 }}
            className="w-full"
          >
            {/* Main Result Card */}
            <div className="gradient-border mb-6">
              <div className="bg-gradient-card rounded-xl p-6">
                <div className="flex flex-col lg:flex-row gap-6">
                  {/* Image */}
                  <div className="relative flex-shrink-0">
                    <img
                      src={preview}
                      alt={`Analyzed satellite image ${resultIndex + 1}`}
                      className="w-full lg:w-64 h-64 object-cover rounded-lg"
                    />
                    <div
                      className="absolute bottom-3 left-3 px-3 py-1.5 rounded-full text-sm font-medium"
                      style={{
                        backgroundColor: topPrediction?.class?.color || '#gray',
                        color: 'white',
                        textShadow: '0 1px 2px rgba(0,0,0,0.3)'
                      }}
                    >
                      {topPrediction?.class?.name || 'Failed'}
                    </div>
                    <div className="absolute top-3 right-3 bg-black/50 text-white px-2 py-1 rounded text-xs">
                      Image {resultIndex + 1}
                    </div>
                  </div>

                  {/* Top Result */}
                  <div className="flex-1">
                    {result.predictions.length > 0 && topPrediction ? (
                      <>
                        <div className="flex items-center gap-2 mb-2">
                          <TrendingUp className="w-5 h-5 text-primary" />
                          <span className="text-sm text-muted-foreground uppercase tracking-wide">Primary Classification</span>
                        </div>

                        <h3 className="text-3xl font-bold mb-2" style={{ color: topPrediction.class.color }}>
                          {topPrediction.class.name}
                        </h3>

                        <p className="text-muted-foreground mb-4">
                          {topPrediction.class.description}
                        </p>

                        {/* Confidence */}
                        <div className="mb-6">
                          <div className="flex justify-between items-center mb-2">
                            <span className="text-sm text-muted-foreground">Confidence</span>
                            <span className="text-2xl font-mono font-bold text-primary">
                              {(topPrediction.probability * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="h-3 bg-muted rounded-full overflow-hidden">
                            <motion.div
                              className="h-full rounded-full"
                              style={{ backgroundColor: topPrediction.class.color }}
                              initial={{ width: 0 }}
                              animate={{ width: `${topPrediction.probability * 100}%` }}
                              transition={{ duration: 1.2, ease: "easeOut" }}
                            />
                          </div>
                        </div>

                        {/* Stats */}
                        <div className="flex gap-6 text-sm">
                          <div className="flex items-center gap-2 text-muted-foreground">
                            <Clock className="w-4 h-4" />
                            <span>Processed in {(result.processingTime / 1000).toFixed(2)}s</span>
                          </div>
                          <div className="flex items-center gap-2 text-muted-foreground">
                            <ImageIcon className="w-4 h-4" />
                            <span>{result.imageSize.width}×{result.imageSize.height}px</span>
                          </div>
                        </div>
                      </>
                    ) : (
                      <div className="text-center py-8 text-muted-foreground">
                        <p className="text-lg font-medium">Classification Failed</p>
                        <p className="text-sm">Unable to process this image</p>
                      </div>
                    )}
                  </div>
                </div>
              </div>
            </div>

            {/* All Predictions */}
            <div className="bg-gradient-card rounded-xl border border-border p-6">
              <h4 className="text-lg font-semibold mb-4">All Classifications - Image {resultIndex + 1}</h4>

              {result.predictions.length === 0 ? (
                <div className="text-center py-8 text-muted-foreground">
                  <p>Classification failed for this image</p>
                </div>
              ) : (
                <div className="space-y-3">
                  {result.predictions.map((pred, index) => (
                    <motion.div
                      key={pred.class.id}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="flex items-center gap-4"
                    >
                      {/* Rank */}
                      <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center flex-shrink-0">
                        <span className="text-sm font-mono font-bold">{index + 1}</span>
                      </div>

                      {/* Class Info */}
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <div
                            className="w-3 h-3 rounded-full flex-shrink-0"
                            style={{ backgroundColor: pred.class.color }}
                          />
                          <span className="font-medium truncate">{pred.class.name}</span>
                        </div>

                        {/* Progress Bar */}
                        <div className="h-2 bg-muted rounded-full overflow-hidden">
                          <motion.div
                            className="h-full rounded-full classification-bar"
                            style={{ backgroundColor: pred.class.color }}
                            initial={{ width: 0 }}
                            animate={{ width: `${pred.probability * 100}%` }}
                            transition={{ duration: 1, delay: index * 0.1, ease: "easeOut" }}
                          />
                        </div>
                      </div>

                      {/* Percentage */}
                      <div className="w-16 text-right">
                        <span className="font-mono text-sm">
                          {(pred.probability * 100).toFixed(1)}%
                        </span>
                      </div>
                    </motion.div>
                  ))}
                </div>
              )}
            </div>
          </motion.div>
        );
      })}
    </motion.div>
  );
}
