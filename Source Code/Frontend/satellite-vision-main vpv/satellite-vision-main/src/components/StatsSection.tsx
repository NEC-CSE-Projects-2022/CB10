import { motion } from 'framer-motion';
import { Target, Zap, Database, Brain } from 'lucide-react';

const stats = [
  {
    icon: Target,
    value: '97.5%',
    label: 'Accuracy',
    description: 'YOLOv8-nano model',
  },
  {
    icon: Database,
    value: '27K+',
    label: 'Training Images',
    description: 'EuroSAT dataset',
  },
  {
    icon: Brain,
    value: '10',
    label: 'Land Cover Classes',
    description: 'Multi-class classification',
  },
  {
    icon: Zap,
    value: '<2s',
    label: 'Processing Time',
    description: 'Real-time inference',
  },
];

export function StatsSection() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {stats.map((stat, index) => (
        <motion.div
          key={stat.label}
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          transition={{ delay: index * 0.1 }}
          viewport={{ once: true }}
          className="stats-card rounded-xl p-5 text-center"
        >
          <stat.icon className="w-8 h-8 text-primary mx-auto mb-3" />
          <div className="text-3xl font-bold text-gradient-orbital mb-1">
            {stat.value}
          </div>
          <div className="font-medium mb-1">{stat.label}</div>
          <div className="text-xs text-muted-foreground">{stat.description}</div>
        </motion.div>
      ))}
    </div>
  );
}
