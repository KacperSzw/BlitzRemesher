// Local durable JSON artifacts. No cloud configuration or provider imports.
import fs from 'node:fs';
import path from 'node:path';

export const read = (p) => JSON.parse(fs.readFileSync(p, 'utf8'));
export function write(p, value) {
  fs.mkdirSync(path.dirname(p), { recursive: true });
  const temporary = p + `.${process.pid}.part`;
  const fd = fs.openSync(temporary, 'w', 0o600);
  try {
    fs.writeFileSync(fd, JSON.stringify(value, null, 2) + '\n');
    fs.fsyncSync(fd);
  } finally {
    fs.closeSync(fd);
  }
  fs.renameSync(temporary, p);
  const directory = fs.openSync(path.dirname(p), 'r');
  try {
    fs.fsyncSync(directory);
  } finally {
    fs.closeSync(directory);
  }
}
