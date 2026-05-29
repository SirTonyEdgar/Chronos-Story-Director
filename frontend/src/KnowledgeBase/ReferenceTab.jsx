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
      description="📚 Reference Material: Two types — Style references teach the AI how to write (prose rhythm, voice, tone), World references teach it how things work (real-world mechanics, period detail, domain knowledge). To mark a reference as a Style reference, start the title with [Style] — e.g. '[Style] McCarthy sparse prose'. Everything else is treated as a World reference."
      placeholder="e.g. Paste a McCarthy paragraph for prose rhythm reference, or describe how 1930s naval intelligence actually communicated..."
    />
  );
}