import React from 'react';
import SharedEditor from './SharedEditor';
import { BookMarked } from 'lucide-react';

export default function ReferenceTab({ profile }) {
  return (
    <SharedEditor
      profile={profile}
      category="Reference"
      icon={<BookMarked />}
      color="#06b6d4"
      description="📚 Reference Material: Style references (prose examples, voice samples, tonal guides) and world texture references (real-world mechanics, period-accurate detail, domain knowledge). These inform how the AI writes and what it knows about how things actually work."
      placeholder="e.g. Paste a McCarthy paragraph for prose rhythm reference, or describe how 1930s naval intelligence actually communicated..."
    />
  );
}