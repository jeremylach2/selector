// The parsed history, kept for the rest of the tab's session so `/watch`
// can use a visitor's own export after they've dropped it on the landing
// page. Module state only: nothing is stored or sent anywhere.

import type { History } from "./types";

export type SessionHistory = { history: History; source: "sample" | "own" };

let current: SessionHistory | null = null;

export const getSessionHistory = () => current;
export const setSessionHistory = (h: SessionHistory | null) => {
  current = h;
};
